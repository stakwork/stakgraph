using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Filters;

namespace CSharpTestServer.Filters;

public class ValidateModelAttribute : ActionFilterAttribute
{
    public override void OnActionExecuting(ActionExecutingContext context)
    {
        if (!context.ModelState.IsValid)
        {
            context.Result = new BadRequestObjectResult(context.ModelState);
        }
    }
}

public class ApiExceptionFilterAttribute : ExceptionFilterAttribute
{
    private readonly ILogger<ApiExceptionFilterAttribute> _logger;

    public ApiExceptionFilterAttribute(ILogger<ApiExceptionFilterAttribute> logger)
    {
        _logger = logger;
    }

    public override void OnException(ExceptionContext context)
    {
        _logger.LogError(context.Exception, "Unhandled exception in {Controller}.{Action}",
            context.RouteData.Values["controller"],
            context.RouteData.Values["action"]);

        var (statusCode, message) = context.Exception switch
        {
            KeyNotFoundException => (StatusCodes.Status404NotFound, "Resource not found"),
            UnauthorizedAccessException => (StatusCodes.Status401Unauthorized, "Unauthorized"),
            InvalidOperationException => (StatusCodes.Status400BadRequest, context.Exception.Message),
            _ => (StatusCodes.Status500InternalServerError, "An error occurred")
        };

        context.Result = new ObjectResult(new { error = message })
        {
            StatusCode = statusCode
        };

        context.ExceptionHandled = true;
    }
}

public class RequireApiKeyAttribute : ActionFilterAttribute
{
    private const string ApiKeyHeader = "X-API-Key";

    public override void OnActionExecuting(ActionExecutingContext context)
    {
        var configuration = context.HttpContext.RequestServices.GetRequiredService<IConfiguration>();
        var validApiKey = configuration["ApiKey"];

        if (string.IsNullOrEmpty(validApiKey))
        {
            return;
        }

        if (!context.HttpContext.Request.Headers.TryGetValue(ApiKeyHeader, out var providedApiKey))
        {
            context.Result = new UnauthorizedObjectResult(new { error = "API key required" });
            return;
        }

        if (providedApiKey != validApiKey)
        {
            context.Result = new UnauthorizedObjectResult(new { error = "Invalid API key" });
        }
    }
}

public class AuditLogAttribute : ActionFilterAttribute
{
    public override async Task OnActionExecutionAsync(ActionExecutingContext context, ActionExecutionDelegate next)
    {
        var auditService = context.HttpContext.RequestServices.GetService<Services.IAuditService>();
        var userId = GetUserId(context.HttpContext);
        var action = $"{context.RouteData.Values["controller"]}.{context.RouteData.Values["action"]}";

        var result = await next();

        if (auditService != null && userId.HasValue)
        {
            var details = result.Exception == null ? "Success" : $"Failed: {result.Exception.Message}";
            await auditService.LogActionAsync(action, userId.Value, details);
        }
    }

    private int? GetUserId(HttpContext context)
    {
        var claim = context.User.FindFirst("sub");
        if (claim != null && int.TryParse(claim.Value, out var userId))
        {
            return userId;
        }
        return null;
    }
}

public class CacheResponseAttribute : ActionFilterAttribute
{
    private readonly int _durationSeconds;

    public CacheResponseAttribute(int durationSeconds = 60)
    {
        _durationSeconds = durationSeconds;
    }

    public override async Task OnActionExecutionAsync(ActionExecutingContext context, ActionExecutionDelegate next)
    {
        var cacheService = context.HttpContext.RequestServices.GetService<Services.ICacheService>();
        
        if (cacheService == null || context.HttpContext.Request.Method != "GET")
        {
            await next();
            return;
        }

        var cacheKey = GenerateCacheKey(context);
        var cachedResponse = await cacheService.GetAsync<object>(cacheKey);

        if (cachedResponse != null)
        {
            context.Result = new OkObjectResult(cachedResponse);
            return;
        }

        var executedContext = await next();

        if (executedContext.Result is OkObjectResult { Value: { } value })
        {
            await cacheService.SetAsync(cacheKey, value, TimeSpan.FromSeconds(_durationSeconds));
        }
    }

    private string GenerateCacheKey(ActionExecutingContext context)
    {
        var path = context.HttpContext.Request.Path.ToString();
        var query = context.HttpContext.Request.QueryString.ToString();
        return $"response:{path}{query}";
    }
}

public class ThrottleAttribute : ActionFilterAttribute
{
    private readonly int _seconds;
    private readonly int _maxRequests;
    private static readonly Dictionary<string, ThrottleInfo> _throttles = new();

    public ThrottleAttribute(int seconds = 1, int maxRequests = 1)
    {
        _seconds = seconds;
        _maxRequests = maxRequests;
    }

    public override void OnActionExecuting(ActionExecutingContext context)
    {
        var key = GenerateThrottleKey(context);
        var now = DateTime.UtcNow;

        lock (_throttles)
        {
            if (_throttles.TryGetValue(key, out var info))
            {
                if ((now - info.WindowStart).TotalSeconds < _seconds)
                {
                    if (info.RequestCount >= _maxRequests)
                    {
                        context.Result = new ObjectResult(new { error = "Too many requests" })
                        {
                            StatusCode = StatusCodes.Status429TooManyRequests
                        };
                        return;
                    }
                    info.RequestCount++;
                }
                else
                {
                    info.WindowStart = now;
                    info.RequestCount = 1;
                }
            }
            else
            {
                _throttles[key] = new ThrottleInfo { WindowStart = now, RequestCount = 1 };
            }
        }
    }

    private string GenerateThrottleKey(ActionExecutingContext context)
    {
        var ip = context.HttpContext.Connection.RemoteIpAddress?.ToString() ?? "unknown";
        var action = $"{context.RouteData.Values["controller"]}.{context.RouteData.Values["action"]}";
        return $"{ip}:{action}";
    }

    private class ThrottleInfo
    {
        public DateTime WindowStart { get; set; }
        public int RequestCount { get; set; }
    }
}
