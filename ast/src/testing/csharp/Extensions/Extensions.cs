namespace CSharpTestServer.Extensions;

public static class ServiceCollectionExtensions
{
    public static IServiceCollection AddApplicationServices(this IServiceCollection services)
    {
        services.AddScoped<Services.IPersonService, Services.PersonService>();
        services.AddScoped<Services.IArticleService, Services.ArticleService>();
        services.AddScoped<Services.IOrderService, Services.OrderService>();
        return services;
    }

    public static IServiceCollection AddRepositories(this IServiceCollection services)
    {
        services.AddScoped<Repositories.IPersonRepository, Repositories.PersonRepository>();
        services.AddScoped<Repositories.IArticleRepository, Repositories.ArticleRepository>();
        services.AddScoped<Repositories.IOrderRepository, Repositories.OrderRepository>();
        services.AddScoped<Repositories.IProductRepository, Repositories.ProductRepository>();
        return services;
    }

    public static IServiceCollection AddInfrastructureServices(this IServiceCollection services)
    {
        services.AddScoped<Services.ICacheService, Services.RedisCacheService>();
        services.AddScoped<Services.IFileStorageService, Services.S3FileStorageService>();
        services.AddScoped<Services.IEmailService, Services.EmailService>();
        return services;
    }
}

public static class StringExtensions
{
    public static string ToSlug(this string value)
    {
        return value.ToLower()
            .Replace(" ", "-")
            .Replace(".", "")
            .Replace(",", "")
            .Replace("'", "")
            .Replace("\"", "");
    }

    public static string Truncate(this string value, int maxLength)
    {
        if (string.IsNullOrEmpty(value) || value.Length <= maxLength)
        {
            return value;
        }
        return value[..maxLength] + "...";
    }

    public static bool IsValidEmail(this string value)
    {
        try
        {
            var addr = new System.Net.Mail.MailAddress(value);
            return addr.Address == value;
        }
        catch
        {
            return false;
        }
    }
}

public static class DateTimeExtensions
{
    public static string ToRelativeTime(this DateTime dateTime)
    {
        var ts = DateTime.UtcNow - dateTime;

        return ts.TotalSeconds switch
        {
            < 60 => "just now",
            < 120 => "a minute ago",
            < 3600 => $"{ts.Minutes} minutes ago",
            < 7200 => "an hour ago",
            < 86400 => $"{ts.Hours} hours ago",
            < 172800 => "yesterday",
            < 604800 => $"{ts.Days} days ago",
            < 1209600 => "a week ago",
            < 2592000 => $"{ts.Days / 7} weeks ago",
            < 5184000 => "a month ago",
            _ => $"{ts.Days / 30} months ago"
        };
    }

    public static DateTime StartOfDay(this DateTime dateTime)
    {
        return dateTime.Date;
    }

    public static DateTime EndOfDay(this DateTime dateTime)
    {
        return dateTime.Date.AddDays(1).AddTicks(-1);
    }

    public static DateTime StartOfWeek(this DateTime dateTime, DayOfWeek startOfWeek = DayOfWeek.Monday)
    {
        int diff = (7 + (dateTime.DayOfWeek - startOfWeek)) % 7;
        return dateTime.AddDays(-1 * diff).Date;
    }

    public static DateTime StartOfMonth(this DateTime dateTime)
    {
        return new DateTime(dateTime.Year, dateTime.Month, 1);
    }
}

public static class EnumerableExtensions
{
    public static IEnumerable<T> WhereIf<T>(this IEnumerable<T> source, bool condition, Func<T, bool> predicate)
    {
        return condition ? source.Where(predicate) : source;
    }

    public static (IEnumerable<T> Matched, IEnumerable<T> Unmatched) Partition<T>(
        this IEnumerable<T> source, Func<T, bool> predicate)
    {
        var grouped = source.GroupBy(predicate).ToDictionary(g => g.Key, g => g.AsEnumerable());
        return (
            grouped.GetValueOrDefault(true, Enumerable.Empty<T>()),
            grouped.GetValueOrDefault(false, Enumerable.Empty<T>())
        );
    }

    public static async Task<List<TResult>> SelectAsync<T, TResult>(
        this IEnumerable<T> source, Func<T, Task<TResult>> selector)
    {
        var results = new List<TResult>();
        foreach (var item in source)
        {
            results.Add(await selector(item));
        }
        return results;
    }
}

public static class HttpContextExtensions
{
    public static int? GetCurrentUserId(this HttpContext context)
    {
        var claim = context.User.FindFirst("sub");
        if (claim != null && int.TryParse(claim.Value, out var userId))
        {
            return userId;
        }
        return null;
    }

    public static string GetClientIp(this HttpContext context)
    {
        var forwardedFor = context.Request.Headers["X-Forwarded-For"].FirstOrDefault();
        if (!string.IsNullOrEmpty(forwardedFor))
        {
            return forwardedFor.Split(',')[0].Trim();
        }
        return context.Connection.RemoteIpAddress?.ToString() ?? "unknown";
    }

    public static string? GetCorrelationId(this HttpContext context)
    {
        return context.Items["CorrelationId"]?.ToString();
    }
}

public static class QueryableExtensions
{
    public static IQueryable<T> Paginate<T>(this IQueryable<T> query, int page, int pageSize)
    {
        return query
            .Skip((page - 1) * pageSize)
            .Take(pageSize);
    }

    public static IQueryable<T> WhereIf<T>(
        this IQueryable<T> query, bool condition, System.Linq.Expressions.Expression<Func<T, bool>> predicate)
    {
        return condition ? query.Where(predicate) : query;
    }
}
