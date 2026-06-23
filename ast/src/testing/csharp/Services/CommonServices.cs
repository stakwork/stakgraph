// @ast node: Class "EmailService"
// @ast node: Class "IAdminService"
// @ast node: Class "IAuditService"
// @ast node: Class "IAuthService"
// @ast node: Class "ICacheService"
// @ast edge: ParentOf -> Class "RedisCacheService" "CommonServices.cs"
// @ast node: Class "IEmailService"
// @ast edge: ParentOf -> Class "EmailService" "CommonServices.cs"
// @ast node: Class "IFileStorageService"
// @ast edge: ParentOf -> Class "S3FileStorageService" "CommonServices.cs"
// @ast node: Class "INotificationService"
// @ast edge: ParentOf -> Class "NotificationService" "CommonServices.cs"
// @ast node: Class "IPaymentService"
// @ast edge: ParentOf -> Class "PaymentService" "CommonServices.cs"
// @ast node: Class "IProductService"
// @ast node: Class "IPushNotificationService"
// @ast node: Class "IReportService"
// @ast node: Class "IUserService"
// @ast node: Class "NotificationService"
// @ast node: Class "PaymentService"
// @ast node: Class "RedisCacheService"
// @ast node: Class "S3FileStorageService"
// @ast node: Trait "IAdminService"
// @ast node: Trait "IAuditService"
// @ast node: Trait "IAuthService"
// @ast node: Trait "ICacheService"
// @ast node: Trait "IEmailService"
// @ast node: Trait "IFileStorageService"
// @ast node: Trait "INotificationService"
// @ast node: Trait "IPaymentService"
// @ast node: Trait "IProductService"
// @ast node: Trait "IPushNotificationService"
// @ast node: Trait "IReportService"
// @ast node: Trait "IUserService"
// @ast node: Function "AddImageAsync"
// @ast node: Function "AddReviewAsync"
// @ast node: Function "AssignRoleAsync"
// @ast node: Function "BanAsync"
// @ast node: Function "BroadcastAsync"
// @ast node: Function "ChangePasswordAsync"
// @ast node: Function "CreateAsync"
// @ast node: Function "DeleteAsync"
// @ast node: Function "DeleteAsync"
// @ast node: Function "DeleteAsync"
// @ast node: Function "DisableMaintenanceModeAsync"
// @ast node: Function "DownloadAsync"
// @ast node: Function "DownloadAsync"
// @ast node: Function "EmailService"
// @ast node: Function "EnableMaintenanceModeAsync"
// @ast node: Function "GeneratePasswordResetTokenAsync"
// @ast node: Function "GenerateTwoFactorSetupAsync"
// @ast node: Function "GetAllAsync"
// @ast node: Function "GetAllAsync"
// @ast node: Function "GetAllAsync"
// @ast node: Function "GetAsync"
// @ast node: Function "GetAsync"
// @ast node: Function "GetByCategoryAsync"
// @ast node: Function "GetByIdAsync"
// @ast node: Function "GetByIdAsync"
// @ast node: Function "GetByIdAsync"
// @ast node: Function "GetDashboardAsync"
// @ast node: Function "GetLogsAsync"
// @ast node: Function "GetReviewsAsync"
// @ast node: Function "GetSettingsAsync"
// @ast node: Function "GetStatsAsync"
// @ast node: Function "InvalidateAsync"
// @ast node: Function "InvalidateAsync"
// @ast node: Function "InvalidatePatternAsync"
// @ast node: Function "InvalidatePatternAsync"
// @ast node: Function "LogActionAsync"
// @ast node: Function "LoginAsync"
// @ast node: Function "LogoutAsync"
// @ast node: Function "NotificationService"
// @ast node: Function "PaymentService"
// @ast node: Function "ProcessPaymentAsync"
// @ast node: Function "ProcessPaymentAsync"
// @ast node: Function "RefreshTokenAsync"
// @ast node: Function "RefundAsync"
// @ast node: Function "RefundAsync"
// @ast node: Function "RegisterAsync"
// @ast node: Function "RemoveImageAsync"
// @ast node: Function "RemoveRoleAsync"
// @ast node: Function "ResetPasswordAsync"
// @ast node: Function "ResolveAsync"
// @ast node: Function "S3FileStorageService"
// @ast node: Function "SearchAsync"
// @ast node: Function "SendAsync"
// @ast node: Function "SendAsync"
// @ast node: Function "SendAsync"
// @ast node: Function "SendOrderCancelledAsync"
// @ast node: Function "SendOrderCancelledAsync"
// @ast node: Function "SendOrderConfirmationAsync"
// @ast node: Function "SendOrderConfirmationAsync"
// @ast node: Function "SendPasswordResetAsync"
// @ast node: Function "SendPasswordResetAsync"
// @ast node: Function "SendPasswordResetEmailAsync"
// @ast node: Function "SendPasswordResetEmailAsync"
// @ast node: Function "SendPushNotificationAsync"
// @ast node: Function "SendPushNotificationAsync"
// @ast node: Function "SendVerificationEmailAsync"
// @ast node: Function "SendVerificationEmailAsync"
// @ast node: Function "SendWelcomeEmailAsync"
// @ast node: Function "SendWelcomeEmailAsync"
// @ast node: Function "SetAsync"
// @ast node: Function "SetAsync"
// @ast node: Function "UnbanAsync"
// @ast node: Function "UpdateAsync"
// @ast node: Function "UpdateInventoryAsync"
// @ast node: Function "UpdateSettingsAsync"
// @ast node: Function "UploadAsync"
// @ast node: Function "UploadAsync"
// @ast node: Function "VerifyAndEnableTwoFactorAsync"
// @ast node: Function "VerifyEmailAsync"
// @ast node: Var "_cache"
// @ast node: Var "_emailService"
// @ast node: Var "_logger"
// @ast node: Var "_logger"
// @ast node: Var "_logger"
// @ast node: Var "_logger"
// @ast node: Var "_pushService"
// @ast node: Import "import-imports-srctestingcsharpservicescommonservicescs-121"
using CSharpTestServer.DTOs;

namespace CSharpTestServer.Services;

public interface INotificationService
{
    Task SendWelcomeEmailAsync(string email);
    Task SendOrderConfirmationAsync(string email, Guid orderId);
    Task SendOrderCancelledAsync(string email, Guid orderId);
    Task SendPasswordResetAsync(string email, string token);
    Task SendPushNotificationAsync(int userId, string title, string message);
}

public class NotificationService : INotificationService
{
    private readonly IEmailService _emailService;
    private readonly IPushNotificationService _pushService;
    private readonly ILogger<NotificationService> _logger;

    public NotificationService(
        IEmailService emailService,
        IPushNotificationService pushService,
        ILogger<NotificationService> logger)
    {
        _emailService = emailService;
        _pushService = pushService;
        _logger = logger;
    }

    public async Task SendWelcomeEmailAsync(string email)
    {
        await _emailService.SendAsync(email, "Welcome!", "Welcome to our platform!");
        _logger.LogInformation("Welcome email sent to {Email}", email);
    }

    public async Task SendOrderConfirmationAsync(string email, Guid orderId)
    {
        await _emailService.SendAsync(email, "Order Confirmed", $"Your order {orderId} has been confirmed.");
        _logger.LogInformation("Order confirmation sent to {Email} for order {OrderId}", email, orderId);
    }

    public async Task SendOrderCancelledAsync(string email, Guid orderId)
    {
        await _emailService.SendAsync(email, "Order Cancelled", $"Your order {orderId} has been cancelled.");
        _logger.LogInformation("Order cancellation sent to {Email} for order {OrderId}", email, orderId);
    }

    public async Task SendPasswordResetAsync(string email, string token)
    {
        var resetLink = $"https://example.com/reset-password?token={token}";
        await _emailService.SendAsync(email, "Reset Password", $"Click here to reset: {resetLink}");
    }

    public async Task SendPushNotificationAsync(int userId, string title, string message)
    {
        await _pushService.SendAsync(userId, title, message);
    }
}

public interface IEmailService
{
    Task SendAsync(string to, string subject, string body);
    Task SendVerificationEmailAsync(string email, string token);
    Task SendPasswordResetEmailAsync(string email, string token);
}

public class EmailService : IEmailService
{
    private readonly ILogger<EmailService> _logger;

    public EmailService(ILogger<EmailService> logger)
    {
        _logger = logger;
    }

    public async Task SendAsync(string to, string subject, string body)
    {
        _logger.LogInformation("Sending email to {To}: {Subject}", to, subject);
        await Task.Delay(100);
    }

    public async Task SendVerificationEmailAsync(string email, string token)
    {
        var link = $"https://example.com/verify?token={token}";
        await SendAsync(email, "Verify your email", $"Click to verify: {link}");
    }

    public async Task SendPasswordResetEmailAsync(string email, string token)
    {
        var link = $"https://example.com/reset?token={token}";
        await SendAsync(email, "Reset your password", $"Click to reset: {link}");
    }
}

public interface IPushNotificationService
{
    Task SendAsync(int userId, string title, string message);
}

public interface IPaymentService
{
    Task<PaymentResultDto> ProcessPaymentAsync(PaymentRequest request);
    Task<RefundResultDto> RefundAsync(Guid orderId);
}

public class PaymentService : IPaymentService
{
    private readonly ILogger<PaymentService> _logger;

    public PaymentService(ILogger<PaymentService> logger)
    {
        _logger = logger;
    }

    public async Task<PaymentResultDto> ProcessPaymentAsync(PaymentRequest request)
    {
        _logger.LogInformation("Processing payment for order {OrderId}", request.OrderId);
        await Task.Delay(100);

        return new PaymentResultDto
        {
            Success = true,
            TransactionId = Guid.NewGuid().ToString(),
            Message = "Payment processed successfully"
        };
    }

    public async Task<RefundResultDto> RefundAsync(Guid orderId)
    {
        _logger.LogInformation("Processing refund for order {OrderId}", orderId);
        await Task.Delay(100);

        return new RefundResultDto
        {
            Success = true,
            RefundId = Guid.NewGuid().ToString(),
            Message = "Refund processed successfully"
        };
    }
}

public interface ICacheService
{
    Task<T?> GetAsync<T>(string key);
    Task SetAsync<T>(string key, T value, TimeSpan? expiry = null);
    Task InvalidateAsync(string key);
    Task InvalidatePatternAsync(string pattern);
}

public class RedisCacheService : ICacheService
{
    private readonly Dictionary<string, object> _cache = new();

    public Task<T?> GetAsync<T>(string key)
    {
        if (_cache.TryGetValue(key, out var value))
        {
            return Task.FromResult((T?)value);
        }
        return Task.FromResult(default(T));
    }

    public Task SetAsync<T>(string key, T value, TimeSpan? expiry = null)
    {
        _cache[key] = value!;
        return Task.CompletedTask;
    }

    public Task InvalidateAsync(string key)
    {
        _cache.Remove(key);
        return Task.CompletedTask;
    }

    public Task InvalidatePatternAsync(string pattern)
    {
        var keysToRemove = _cache.Keys
            .Where(k => k.StartsWith(pattern.TrimEnd('*')))
            .ToList();
        
        foreach (var key in keysToRemove)
        {
            _cache.Remove(key);
        }
        
        return Task.CompletedTask;
    }
}

public interface IFileStorageService
{
    Task<string> UploadAsync(IFormFile file, string path);
    Task DeleteAsync(string path);
    Task<Stream> DownloadAsync(string path);
}

public class S3FileStorageService : IFileStorageService
{
    private readonly ILogger<S3FileStorageService> _logger;

    public S3FileStorageService(ILogger<S3FileStorageService> logger)
    {
        _logger = logger;
    }

    public async Task<string> UploadAsync(IFormFile file, string path)
    {
        _logger.LogInformation("Uploading file to {Path}", path);
        await Task.Delay(100);
        return $"https://s3.amazonaws.com/bucket/{path}/{file.FileName}";
    }

    public async Task DeleteAsync(string path)
    {
        _logger.LogInformation("Deleting file at {Path}", path);
        await Task.Delay(50);
    }

    public Task<Stream> DownloadAsync(string path)
    {
        _logger.LogInformation("Downloading file from {Path}", path);
        return Task.FromResult<Stream>(new MemoryStream());
    }
}

public interface IAuthService
{
    Task<AuthResponseDto> RegisterAsync(RegisterRequest request);
    Task<AuthResponseDto> LoginAsync(LoginRequest request);
    Task LogoutAsync(string token);
    Task<AuthResponseDto> RefreshTokenAsync(string refreshToken);
    Task<string?> GeneratePasswordResetTokenAsync(string email);
    Task<bool> ResetPasswordAsync(string token, string newPassword);
    Task<bool> VerifyEmailAsync(string token);
    Task<bool> ChangePasswordAsync(int userId, string currentPassword, string newPassword);
    Task<TwoFactorSetupDto> GenerateTwoFactorSetupAsync(int userId);
    Task<bool> VerifyAndEnableTwoFactorAsync(int userId, string code);
}

public interface IProductService
{
    Task<PagedResult<ProductDto>> GetAllAsync(ProductQueryParams queryParams);
    Task<ProductDto?> GetByIdAsync(int id);
    Task<IEnumerable<ProductDto>> GetByCategoryAsync(int categoryId);
    Task<IEnumerable<ProductDto>> SearchAsync(string query, decimal? minPrice, decimal? maxPrice);
    Task<ProductDto> CreateAsync(CreateProductRequest request);
    Task<ProductDto> UpdateAsync(int id, UpdateProductRequest request);
    Task DeleteAsync(int id);
    Task AddImageAsync(int id, string url);
    Task RemoveImageAsync(int id, int imageId);
    Task<IEnumerable<ReviewDto>> GetReviewsAsync(int id);
    Task<ReviewDto> AddReviewAsync(int id, CreateReviewRequest request);
    Task<ProductDto> UpdateInventoryAsync(int id, int quantity);
}

public interface IAdminService
{
    Task<DashboardDto> GetDashboardAsync();
    Task<StatsDto> GetStatsAsync(DateTime startDate, DateTime endDate);
    Task BroadcastAsync(string title, string message, IEnumerable<string> targetRoles);
    Task<SystemSettingsDto> GetSettingsAsync();
    Task<SystemSettingsDto> UpdateSettingsAsync(UpdateSettingsRequest request);
    Task EnableMaintenanceModeAsync(string message, TimeSpan? duration);
    Task DisableMaintenanceModeAsync();
}

public interface IUserService
{
    Task<PagedResult<UserDto>> GetAllAsync(UserQueryParams queryParams);
    Task<UserDto?> GetByIdAsync(int id);
    Task BanAsync(int id, string reason, TimeSpan? duration);
    Task UnbanAsync(int id);
    Task AssignRoleAsync(int id, string role);
    Task RemoveRoleAsync(int id, string role);
}

public interface IReportService
{
    Task<IEnumerable<ReportDto>> GetAllAsync(ReportQueryParams queryParams);
    Task<ReportDto?> GetByIdAsync(int id);
    Task ResolveAsync(int id, string resolution, string? notes);
}

public interface IAuditService
{
    Task LogActionAsync(string action, int userId, string? details);
    Task<PagedResult<AuditLogDto>> GetLogsAsync(AuditLogQueryParams queryParams);
}
