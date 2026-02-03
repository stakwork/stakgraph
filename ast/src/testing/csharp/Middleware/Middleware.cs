using System.Diagnostics;
using System.Text.Json;

namespace CSharpTestServer.Middleware;

public class RequestLoggingMiddleware
{
    private readonly RequestDelegate _next;
    private readonly ILogger<RequestLoggingMiddleware> _logger;

    public RequestLoggingMiddleware(RequestDelegate next, ILogger<RequestLoggingMiddleware> logger)
    {
        _next = next;
        _logger = logger;
    }

    public async Task InvokeAsync(HttpContext context)
    {
        var sw = Stopwatch.StartNew();
        var requestId = Guid.NewGuid().ToString("N")[..8];

        context.Items["RequestId"] = requestId;

        _logger.LogInformation(
            "[{RequestId}] {Method} {Path} started",
            requestId,
            context.Request.Method,
            context.Request.Path);

        try
        {
            await _next(context);
        }
        finally
        {
            sw.Stop();
            _logger.LogInformation(
                "[{RequestId}] {Method} {Path} completed with {StatusCode} in {ElapsedMs}ms",
                requestId,
                context.Request.Method,
                context.Request.Path,
                context.Response.StatusCode,
                sw.ElapsedMilliseconds);
        }
    }
}

public class ExceptionHandlingMiddleware
{
    private readonly RequestDelegate _next;
    private readonly ILogger<ExceptionHandlingMiddleware> _logger;
    private readonly IHostEnvironment _env;

    public ExceptionHandlingMiddleware(
        RequestDelegate next,
        ILogger<ExceptionHandlingMiddleware> logger,
        IHostEnvironment env)
    {
        _next = next;
        _logger = logger;
        _env = env;
    }

    public async Task InvokeAsync(HttpContext context)
    {
        try
        {
            await _next(context);
        }
        catch (Exception ex)
        {
            await HandleExceptionAsync(context, ex);
        }
    }

    private async Task HandleExceptionAsync(HttpContext context, Exception exception)
    {
        var requestId = context.Items["RequestId"]?.ToString() ?? "unknown";

        _logger.LogError(exception, "[{RequestId}] Unhandled exception occurred", requestId);

        context.Response.ContentType = "application/json";

        var (statusCode, message) = exception switch
        {
            KeyNotFoundException => (StatusCodes.Status404NotFound, "Resource not found"),
            UnauthorizedAccessException => (StatusCodes.Status401Unauthorized, "Unauthorized"),
            InvalidOperationException => (StatusCodes.Status400BadRequest, exception.Message),
            ArgumentException => (StatusCodes.Status400BadRequest, exception.Message),
            _ => (StatusCodes.Status500InternalServerError, "An unexpected error occurred")
        };

        context.Response.StatusCode = statusCode;

        var response = new
        {
            error = new
            {
                message,
                requestId,
                details = _env.IsDevelopment() ? exception.ToString() : null
            }
        };

        await context.Response.WriteAsync(JsonSerializer.Serialize(response));
    }
}

public class RateLimitingMiddleware
{
    private readonly RequestDelegate _next;
    private readonly ILogger<RateLimitingMiddleware> _logger;
    private readonly Dictionary<string, RateLimitInfo> _clients = new();
    private readonly int _requestsPerMinute;

    public RateLimitingMiddleware(RequestDelegate next, ILogger<RateLimitingMiddleware> logger, int requestsPerMinute = 100)
    {
        _next = next;
        _logger = logger;
        _requestsPerMinute = requestsPerMinute;
    }

    public async Task InvokeAsync(HttpContext context)
    {
        var clientIp = context.Connection.RemoteIpAddress?.ToString() ?? "unknown";

        if (!_clients.TryGetValue(clientIp, out var info))
        {
            info = new RateLimitInfo { WindowStart = DateTime.UtcNow, RequestCount = 0 };
            _clients[clientIp] = info;
        }

        if (DateTime.UtcNow - info.WindowStart > TimeSpan.FromMinutes(1))
        {
            info.WindowStart = DateTime.UtcNow;
            info.RequestCount = 0;
        }

        info.RequestCount++;

        if (info.RequestCount > _requestsPerMinute)
        {
            _logger.LogWarning("Rate limit exceeded for {ClientIp}", clientIp);
            context.Response.StatusCode = StatusCodes.Status429TooManyRequests;
            await context.Response.WriteAsync("Rate limit exceeded. Please try again later.");
            return;
        }

        context.Response.Headers.Append("X-RateLimit-Limit", _requestsPerMinute.ToString());
        context.Response.Headers.Append("X-RateLimit-Remaining", (_requestsPerMinute - info.RequestCount).ToString());

        await _next(context);
    }

    private class RateLimitInfo
    {
        public DateTime WindowStart { get; set; }
        public int RequestCount { get; set; }
    }
}

public class MaintenanceModeMiddleware
{
    private readonly RequestDelegate _next;
    private readonly IConfiguration _configuration;

    public MaintenanceModeMiddleware(RequestDelegate next, IConfiguration configuration)
    {
        _next = next;
        _configuration = configuration;
    }

    public async Task InvokeAsync(HttpContext context)
    {
        var maintenanceMode = _configuration.GetValue<bool>("MaintenanceMode:Enabled");

        if (maintenanceMode)
        {
            var allowedPaths = new[] { "/health", "/api/admin" };
            var isAllowed = allowedPaths.Any(p => context.Request.Path.StartsWithSegments(p));

            if (!isAllowed)
            {
                context.Response.StatusCode = StatusCodes.Status503ServiceUnavailable;
                var message = _configuration["MaintenanceMode:Message"] ?? "Service is under maintenance";
                await context.Response.WriteAsJsonAsync(new { message });
                return;
            }
        }

        await _next(context);
    }
}

public class CorrelationIdMiddleware
{
    private readonly RequestDelegate _next;
    private const string CorrelationIdHeader = "X-Correlation-ID";

    public CorrelationIdMiddleware(RequestDelegate next)
    {
        _next = next;
    }

    public async Task InvokeAsync(HttpContext context)
    {
        var correlationId = context.Request.Headers[CorrelationIdHeader].FirstOrDefault()
            ?? Guid.NewGuid().ToString();

        context.Items["CorrelationId"] = correlationId;
        context.Response.Headers.Append(CorrelationIdHeader, correlationId);

        using (_logger?.BeginScope(new Dictionary<string, object>
        {
            ["CorrelationId"] = correlationId
        }))
        {
            await _next(context);
        }
    }

    private ILogger<CorrelationIdMiddleware>? _logger;
}
