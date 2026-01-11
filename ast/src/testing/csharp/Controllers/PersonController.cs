using Microsoft.AspNetCore.Mvc;
using CSharpTestServer.DTOs;
using CSharpTestServer.Services;

namespace CSharpTestServer.Controllers;

[ApiController]
[Route("api/[controller]")]
public class PersonController : ControllerBase
{
    private readonly IPersonService _personService;
    private readonly INotificationService _notificationService;
    private readonly ILogger<PersonController> _logger;

    public PersonController(
        IPersonService personService,
        INotificationService notificationService,
        ILogger<PersonController> logger)
    {
        _personService = personService;
        _notificationService = notificationService;
        _logger = logger;
    }

    [HttpGet]
    public async Task<ActionResult<IEnumerable<PersonDto>>> GetAll([FromQuery] PaginationParams pagination)
    {
        _logger.LogInformation("Getting all persons with pagination");
        var persons = await _personService.GetAllAsync(pagination.Page, pagination.PageSize);
        return Ok(persons);
    }

    [HttpGet("{id:int}")]
    public async Task<ActionResult<PersonDto>> GetById(int id)
    {
        var person = await _personService.GetByIdAsync(id);
        if (person == null)
        {
            return NotFound(new ErrorResponse { Message = $"Person with id {id} not found" });
        }
        return Ok(person);
    }

    [HttpGet("search")]
    public async Task<ActionResult<IEnumerable<PersonDto>>> Search([FromQuery] string query)
    {
        var results = await _personService.SearchAsync(query);
        return Ok(results);
    }

    [HttpPost]
    public async Task<ActionResult<PersonDto>> Create([FromBody] CreatePersonRequest request)
    {
        if (!ModelState.IsValid)
        {
            return BadRequest(ModelState);
        }

        var person = await _personService.CreateAsync(request);
        await _notificationService.SendWelcomeEmailAsync(person.Email);
        
        return CreatedAtAction(nameof(GetById), new { id = person.Id }, person);
    }

    [HttpPut("{id:int}")]
    public async Task<ActionResult<PersonDto>> Update(int id, [FromBody] UpdatePersonRequest request)
    {
        var person = await _personService.UpdateAsync(id, request);
        if (person == null)
        {
            return NotFound();
        }
        return Ok(person);
    }

    [HttpPatch("{id:int}")]
    public async Task<ActionResult<PersonDto>> PartialUpdate(int id, [FromBody] PatchPersonRequest request)
    {
        var person = await _personService.PartialUpdateAsync(id, request);
        return Ok(person);
    }

    [HttpDelete("{id:int}")]
    public async Task<ActionResult> Delete(int id)
    {
        var result = await _personService.DeleteAsync(id);
        if (!result)
        {
            return NotFound();
        }
        return NoContent();
    }

    [HttpGet("{id:int}/articles")]
    public async Task<ActionResult<IEnumerable<ArticleDto>>> GetPersonArticles(int id)
    {
        var articles = await _personService.GetArticlesAsync(id);
        return Ok(articles);
    }

    [HttpPost("{id:int}/avatar")]
    public async Task<ActionResult<string>> UploadAvatar(int id, IFormFile file)
    {
        var avatarUrl = await _personService.UploadAvatarAsync(id, file);
        return Ok(new { Url = avatarUrl });
    }
}
