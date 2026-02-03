using System.ComponentModel.DataAnnotations;

namespace CSharpTestServer.DTOs;

public class RegisterRequest
{
    [Required]
    [EmailAddress]
    public string Email { get; set; } = "";

    [Required]
    [MinLength(8)]
    public string Password { get; set; } = "";

    [Required]
    public string FirstName { get; set; } = "";

    [Required]
    public string LastName { get; set; } = "";
}

public class LoginRequest
{
    [Required]
    [EmailAddress]
    public string Email { get; set; } = "";

    [Required]
    public string Password { get; set; } = "";

    public bool RememberMe { get; set; }
}

public class AuthResponseDto
{
    public bool Success { get; set; }
    public string? AccessToken { get; set; }
    public string? RefreshToken { get; set; }
    public DateTime? ExpiresAt { get; set; }
    public string? Error { get; set; }
    public string? VerificationToken { get; set; }
    public UserInfo? User { get; set; }
}

public class UserInfo
{
    public int Id { get; set; }
    public string Email { get; set; } = "";
    public string FullName { get; set; } = "";
    public IEnumerable<string> Roles { get; set; } = Array.Empty<string>();
}

public class RefreshTokenRequest
{
    [Required]
    public string RefreshToken { get; set; } = "";
}

public class ForgotPasswordRequest
{
    [Required]
    [EmailAddress]
    public string Email { get; set; } = "";
}

public class ResetPasswordRequest
{
    [Required]
    public string Token { get; set; } = "";

    [Required]
    [MinLength(8)]
    public string NewPassword { get; set; } = "";
}

public class VerifyEmailRequest
{
    [Required]
    public string Token { get; set; } = "";
}

public class ChangePasswordRequest
{
    [Required]
    public string CurrentPassword { get; set; } = "";

    [Required]
    [MinLength(8)]
    public string NewPassword { get; set; } = "";
}

public class TwoFactorSetupDto
{
    public string SecretKey { get; set; } = "";
    public string QrCodeUri { get; set; } = "";
    public IEnumerable<string> RecoveryCodes { get; set; } = Array.Empty<string>();
}

public class VerifyTwoFactorRequest
{
    [Required]
    [StringLength(6, MinimumLength = 6)]
    public string Code { get; set; } = "";
}

public class UserDto
{
    public int Id { get; set; }
    public string Email { get; set; } = "";
    public string FirstName { get; set; } = "";
    public string LastName { get; set; } = "";
    public string FullName { get; set; } = "";
    public bool IsActive { get; set; }
    public bool IsBanned { get; set; }
    public DateTime? BannedUntil { get; set; }
    public string? BanReason { get; set; }
    public bool EmailVerified { get; set; }
    public bool TwoFactorEnabled { get; set; }
    public IEnumerable<string> Roles { get; set; } = Array.Empty<string>();
    public DateTime CreatedAt { get; set; }
    public DateTime? LastLoginAt { get; set; }
}

public class UserQueryParams
{
    public int Page { get; set; } = 1;
    public int PageSize { get; set; } = 20;
    public string? Search { get; set; }
    public string? Role { get; set; }
    public bool? IsActive { get; set; }
    public bool? IsBanned { get; set; }
}

public class BanUserRequest
{
    [Required]
    public string Reason { get; set; } = "";

    public TimeSpan? Duration { get; set; }
}

public class AssignRoleRequest
{
    [Required]
    public string Role { get; set; } = "";
}

public class DashboardDto
{
    public int TotalUsers { get; set; }
    public int TotalOrders { get; set; }
    public decimal TotalRevenue { get; set; }
    public int TotalProducts { get; set; }
    public int PendingOrders { get; set; }
    public int LowStockProducts { get; set; }
    public IEnumerable<RecentOrderDto> RecentOrders { get; set; } = Array.Empty<RecentOrderDto>();
    public IEnumerable<TopProductDto> TopProducts { get; set; } = Array.Empty<TopProductDto>();
}

public class RecentOrderDto
{
    public Guid Id { get; set; }
    public string OrderNumber { get; set; } = "";
    public string CustomerName { get; set; } = "";
    public decimal Amount { get; set; }
    public string Status { get; set; } = "";
    public DateTime CreatedAt { get; set; }
}

public class TopProductDto
{
    public int Id { get; set; }
    public string Name { get; set; } = "";
    public int SalesCount { get; set; }
    public decimal Revenue { get; set; }
}

public class StatsDto
{
    public decimal TotalRevenue { get; set; }
    public int OrderCount { get; set; }
    public int NewUsers { get; set; }
    public decimal AverageOrderValue { get; set; }
    public IEnumerable<DailyStatsDto> DailyStats { get; set; } = Array.Empty<DailyStatsDto>();
}

public class DailyStatsDto
{
    public DateOnly Date { get; set; }
    public decimal Revenue { get; set; }
    public int Orders { get; set; }
    public int Visitors { get; set; }
}

public class BroadcastRequest
{
    [Required]
    public string Title { get; set; } = "";

    [Required]
    public string Message { get; set; } = "";

    public IEnumerable<string>? TargetRoles { get; set; }
}

public class SystemSettingsDto
{
    public bool MaintenanceMode { get; set; }
    public string? MaintenanceMessage { get; set; }
    public bool RegistrationEnabled { get; set; }
    public bool EmailVerificationRequired { get; set; }
    public int MaxLoginAttempts { get; set; }
    public int SessionTimeout { get; set; }
    public decimal TaxRate { get; set; }
    public decimal FreeShippingThreshold { get; set; }
}

public class UpdateSettingsRequest
{
    public bool? MaintenanceMode { get; set; }
    public string? MaintenanceMessage { get; set; }
    public bool? RegistrationEnabled { get; set; }
    public bool? EmailVerificationRequired { get; set; }
    public int? MaxLoginAttempts { get; set; }
    public int? SessionTimeout { get; set; }
    public decimal? TaxRate { get; set; }
    public decimal? FreeShippingThreshold { get; set; }
}

public class MaintenanceRequest
{
    public string Message { get; set; } = "";
    public TimeSpan? EstimatedDuration { get; set; }
}

public class ReportDto
{
    public int Id { get; set; }
    public string Type { get; set; } = "";
    public string Reason { get; set; } = "";
    public string? Description { get; set; }
    public int ReporterId { get; set; }
    public string ReporterName { get; set; } = "";
    public int? TargetUserId { get; set; }
    public int? TargetContentId { get; set; }
    public string TargetContentType { get; set; } = "";
    public string Status { get; set; } = "";
    public string? Resolution { get; set; }
    public string? Notes { get; set; }
    public DateTime CreatedAt { get; set; }
    public DateTime? ResolvedAt { get; set; }
}

public class ReportQueryParams
{
    public int Page { get; set; } = 1;
    public int PageSize { get; set; } = 20;
    public string? Status { get; set; }
    public string? Type { get; set; }
}

public class ResolveReportRequest
{
    [Required]
    public string Resolution { get; set; } = "";

    public string? Notes { get; set; }
}

public class AuditLogDto
{
    public int Id { get; set; }
    public string Action { get; set; } = "";
    public int UserId { get; set; }
    public string UserName { get; set; } = "";
    public string? Details { get; set; }
    public string? IpAddress { get; set; }
    public string? UserAgent { get; set; }
    public DateTime Timestamp { get; set; }
}

public class AuditLogQueryParams
{
    public int Page { get; set; } = 1;
    public int PageSize { get; set; } = 50;
    public string? Action { get; set; }
    public int? UserId { get; set; }
    public DateTime? FromDate { get; set; }
    public DateTime? ToDate { get; set; }
}
