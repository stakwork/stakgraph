using Microsoft.AspNetCore.Mvc;
using CSharpTestServer.DTOs;
using CSharpTestServer.Services;

namespace CSharpTestServer.Controllers;

[ApiController]
[Route("api/[controller]")]
public class AuthController : ControllerBase
{
    private readonly IAuthService _authService;
    private readonly IEmailService _emailService;
    private readonly ILogger<AuthController> _logger;

    public AuthController(
        IAuthService authService,
        IEmailService emailService,
        ILogger<AuthController> logger)
    {
        _authService = authService;
        _emailService = emailService;
        _logger = logger;
    }

    [HttpPost("register")]
    public async Task<ActionResult<AuthResponseDto>> Register([FromBody] RegisterRequest request)
    {
        var result = await _authService.RegisterAsync(request);
        if (!result.Success)
        {
            return BadRequest(new ErrorResponse { Message = result.Error });
        }

        await _emailService.SendVerificationEmailAsync(request.Email, result.VerificationToken);
        _logger.LogInformation("User registered: {Email}", request.Email);

        return Ok(result);
    }

    [HttpPost("login")]
    public async Task<ActionResult<AuthResponseDto>> Login([FromBody] LoginRequest request)
    {
        var result = await _authService.LoginAsync(request);
        if (!result.Success)
        {
            _logger.LogWarning("Failed login attempt for: {Email}", request.Email);
            return Unauthorized(new ErrorResponse { Message = "Invalid credentials" });
        }

        return Ok(result);
    }

    [HttpPost("logout")]
    public async Task<ActionResult> Logout()
    {
        var token = GetBearerToken();
        await _authService.LogoutAsync(token);
        return Ok();
    }

    [HttpPost("refresh")]
    public async Task<ActionResult<AuthResponseDto>> RefreshToken([FromBody] RefreshTokenRequest request)
    {
        var result = await _authService.RefreshTokenAsync(request.RefreshToken);
        if (!result.Success)
        {
            return Unauthorized();
        }
        return Ok(result);
    }

    [HttpPost("forgot-password")]
    public async Task<ActionResult> ForgotPassword([FromBody] ForgotPasswordRequest request)
    {
        var token = await _authService.GeneratePasswordResetTokenAsync(request.Email);
        if (token != null)
        {
            await _emailService.SendPasswordResetEmailAsync(request.Email, token);
        }
        return Ok(new { Message = "If the email exists, a reset link has been sent." });
    }

    [HttpPost("reset-password")]
    public async Task<ActionResult> ResetPassword([FromBody] ResetPasswordRequest request)
    {
        var result = await _authService.ResetPasswordAsync(request.Token, request.NewPassword);
        if (!result)
        {
            return BadRequest(new ErrorResponse { Message = "Invalid or expired token" });
        }
        return Ok(new { Message = "Password reset successfully" });
    }

    [HttpPost("verify-email")]
    public async Task<ActionResult> VerifyEmail([FromBody] VerifyEmailRequest request)
    {
        var result = await _authService.VerifyEmailAsync(request.Token);
        if (!result)
        {
            return BadRequest(new ErrorResponse { Message = "Invalid or expired token" });
        }
        return Ok(new { Message = "Email verified successfully" });
    }

    [HttpPost("change-password")]
    public async Task<ActionResult> ChangePassword([FromBody] ChangePasswordRequest request)
    {
        var userId = GetCurrentUserId();
        var result = await _authService.ChangePasswordAsync(userId, request.CurrentPassword, request.NewPassword);
        if (!result)
        {
            return BadRequest(new ErrorResponse { Message = "Current password is incorrect" });
        }
        return Ok();
    }

    [HttpPost("two-factor/enable")]
    public async Task<ActionResult<TwoFactorSetupDto>> EnableTwoFactor()
    {
        var userId = GetCurrentUserId();
        var setup = await _authService.GenerateTwoFactorSetupAsync(userId);
        return Ok(setup);
    }

    [HttpPost("two-factor/verify")]
    public async Task<ActionResult> VerifyTwoFactor([FromBody] VerifyTwoFactorRequest request)
    {
        var userId = GetCurrentUserId();
        var result = await _authService.VerifyAndEnableTwoFactorAsync(userId, request.Code);
        if (!result)
        {
            return BadRequest(new ErrorResponse { Message = "Invalid code" });
        }
        return Ok();
    }

    private string GetBearerToken()
    {
        var auth = Request.Headers["Authorization"].FirstOrDefault();
        return auth?.Replace("Bearer ", "") ?? "";
    }

    private int GetCurrentUserId()
    {
        var claim = User.FindFirst("sub");
        return int.Parse(claim?.Value ?? "0");
    }
}
