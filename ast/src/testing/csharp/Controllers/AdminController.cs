using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Authorization;
using CSharpTestServer.DTOs;
using CSharpTestServer.Services;

namespace CSharpTestServer.Controllers;

[ApiController]
[Route("api/admin")]
[Authorize(Policy = "AdminOnly")]
public class AdminController : ControllerBase
{
    private readonly IAdminService _adminService;
    private readonly IUserService _userService;
    private readonly IReportService _reportService;
    private readonly IAuditService _auditService;

    public AdminController(
        IAdminService adminService,
        IUserService userService,
        IReportService reportService,
        IAuditService auditService)
    {
        _adminService = adminService;
        _userService = userService;
        _reportService = reportService;
        _auditService = auditService;
    }

    [HttpGet("dashboard")]
    public async Task<ActionResult<DashboardDto>> GetDashboard()
    {
        var dashboard = await _adminService.GetDashboardAsync();
        return Ok(dashboard);
    }

    [HttpGet("users")]
    public async Task<ActionResult<PagedResult<UserDto>>> GetUsers([FromQuery] UserQueryParams queryParams)
    {
        var users = await _userService.GetAllAsync(queryParams);
        return Ok(users);
    }

    [HttpGet("users/{id:int}")]
    public async Task<ActionResult<UserDto>> GetUser(int id)
    {
        var user = await _userService.GetByIdAsync(id);
        if (user == null)
        {
            return NotFound();
        }
        return Ok(user);
    }

    [HttpPost("users/{id:int}/ban")]
    public async Task<ActionResult> BanUser(int id, [FromBody] BanUserRequest request)
    {
        await _userService.BanAsync(id, request.Reason, request.Duration);
        await _auditService.LogActionAsync("UserBanned", id, request.Reason);
        return Ok();
    }

    [HttpPost("users/{id:int}/unban")]
    public async Task<ActionResult> UnbanUser(int id)
    {
        await _userService.UnbanAsync(id);
        await _auditService.LogActionAsync("UserUnbanned", id, null);
        return Ok();
    }

    [HttpPost("users/{id:int}/roles")]
    public async Task<ActionResult> AssignRole(int id, [FromBody] AssignRoleRequest request)
    {
        await _userService.AssignRoleAsync(id, request.Role);
        return Ok();
    }

    [HttpDelete("users/{id:int}/roles/{role}")]
    public async Task<ActionResult> RemoveRole(int id, string role)
    {
        await _userService.RemoveRoleAsync(id, role);
        return Ok();
    }

    [HttpGet("reports")]
    public async Task<ActionResult<IEnumerable<ReportDto>>> GetReports([FromQuery] ReportQueryParams queryParams)
    {
        var reports = await _reportService.GetAllAsync(queryParams);
        return Ok(reports);
    }

    [HttpGet("reports/{id:int}")]
    public async Task<ActionResult<ReportDto>> GetReport(int id)
    {
        var report = await _reportService.GetByIdAsync(id);
        return Ok(report);
    }

    [HttpPost("reports/{id:int}/resolve")]
    public async Task<ActionResult> ResolveReport(int id, [FromBody] ResolveReportRequest request)
    {
        await _reportService.ResolveAsync(id, request.Resolution, request.Notes);
        return Ok();
    }

    [HttpGet("audit-logs")]
    public async Task<ActionResult<PagedResult<AuditLogDto>>> GetAuditLogs([FromQuery] AuditLogQueryParams queryParams)
    {
        var logs = await _auditService.GetLogsAsync(queryParams);
        return Ok(logs);
    }

    [HttpGet("stats")]
    public async Task<ActionResult<StatsDto>> GetStats([FromQuery] DateTime? startDate, [FromQuery] DateTime? endDate)
    {
        var stats = await _adminService.GetStatsAsync(startDate ?? DateTime.UtcNow.AddMonths(-1), endDate ?? DateTime.UtcNow);
        return Ok(stats);
    }

    [HttpPost("broadcast")]
    public async Task<ActionResult> BroadcastNotification([FromBody] BroadcastRequest request)
    {
        await _adminService.BroadcastAsync(request.Title, request.Message, request.TargetRoles);
        return Ok();
    }

    [HttpGet("settings")]
    public async Task<ActionResult<SystemSettingsDto>> GetSettings()
    {
        var settings = await _adminService.GetSettingsAsync();
        return Ok(settings);
    }

    [HttpPut("settings")]
    public async Task<ActionResult<SystemSettingsDto>> UpdateSettings([FromBody] UpdateSettingsRequest request)
    {
        var settings = await _adminService.UpdateSettingsAsync(request);
        return Ok(settings);
    }

    [HttpPost("maintenance/enable")]
    public async Task<ActionResult> EnableMaintenance([FromBody] MaintenanceRequest request)
    {
        await _adminService.EnableMaintenanceModeAsync(request.Message, request.EstimatedDuration);
        return Ok();
    }

    [HttpPost("maintenance/disable")]
    public async Task<ActionResult> DisableMaintenance()
    {
        await _adminService.DisableMaintenanceModeAsync();
        return Ok();
    }
}
