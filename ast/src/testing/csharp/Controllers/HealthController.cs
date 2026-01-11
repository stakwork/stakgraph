using Microsoft.AspNetCore.Mvc;

namespace CSharpTestServer.Controllers;

[ApiController]
[Route("api/[controller]")]
public class HealthController : ControllerBase
{
    private readonly ILogger<HealthController> _logger;

    public HealthController(ILogger<HealthController> logger)
    {
        _logger = logger;
    }

    [HttpGet]
    public ActionResult<HealthStatus> Get()
    {
        return Ok(new HealthStatus
        {
            Status = "healthy",
            Timestamp = DateTime.UtcNow,
            Version = "1.0.0"
        });
    }

    [HttpGet("live")]
    public ActionResult LiveCheck()
    {
        return Ok("alive");
    }

    [HttpGet("ready")]
    public ActionResult<ReadinessStatus> ReadyCheck()
    {
        return Ok(new ReadinessStatus
        {
            Ready = true,
            Database = true,
            Cache = true,
            ExternalServices = true
        });
    }
}

public class HealthStatus
{
    public string Status { get; set; } = "";
    public DateTime Timestamp { get; set; }
    public string Version { get; set; } = "";
}

public class ReadinessStatus
{
    public bool Ready { get; set; }
    public bool Database { get; set; }
    public bool Cache { get; set; }
    public bool ExternalServices { get; set; }
}
