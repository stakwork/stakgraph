// @ast node: Class "StatusController"
// @ast node: Function "GetStatus"
// @ast edge: Calls -> Function "GetDashboardAsync" "CommonServices.cs"
// @ast node: Endpoint "status"
// @ast edge: Handler -> Function "GetStatus" "StatusController.cs"
// @ast node: Import "import-imports-srctestingcsharpcontrollersstatuscontrollercs-7"

using Microsoft.AspNetCore.Mvc;
using CSharpTestServer.Services;

namespace CSharpTestServer.Controllers;

[ApiController]
[Route("api/[controller]")]
public class StatusController(IAdminService adminService) : ControllerBase
{
    [HttpGet("status")]
    public async Task<ActionResult<object>> GetStatus()
    {
        var dashboard = await adminService.GetDashboardAsync();
        return Ok(dashboard);
    }
}
