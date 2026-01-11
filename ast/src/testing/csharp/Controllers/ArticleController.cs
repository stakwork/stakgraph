using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Authorization;
using CSharpTestServer.DTOs;
using CSharpTestServer.Services;

namespace CSharpTestServer.Controllers;

[ApiController]
[Route("api/[controller]")]
[Authorize]
public class ArticleController : ControllerBase
{
    private readonly IArticleService _articleService;
    private readonly IPersonService _personService;
    private readonly ICacheService _cacheService;

    public ArticleController(
        IArticleService articleService,
        IPersonService personService,
        ICacheService cacheService)
    {
        _articleService = articleService;
        _personService = personService;
        _cacheService = cacheService;
    }

    [HttpGet]
    [AllowAnonymous]
    public async Task<ActionResult<PagedResult<ArticleDto>>> GetAll(
        [FromQuery] int page = 1,
        [FromQuery] int pageSize = 10,
        [FromQuery] string? sortBy = null)
    {
        var cacheKey = $"articles:{page}:{pageSize}:{sortBy}";
        var cached = await _cacheService.GetAsync<PagedResult<ArticleDto>>(cacheKey);
        if (cached != null)
        {
            return Ok(cached);
        }

        var articles = await _articleService.GetAllPagedAsync(page, pageSize, sortBy);
        await _cacheService.SetAsync(cacheKey, articles, TimeSpan.FromMinutes(5));
        return Ok(articles);
    }

    [HttpGet("{id:int}")]
    [AllowAnonymous]
    public async Task<ActionResult<ArticleDto>> GetById(int id)
    {
        var article = await _articleService.GetByIdAsync(id);
        if (article == null)
        {
            return NotFound();
        }
        return Ok(article);
    }

    [HttpGet("by-author/{authorId:int}")]
    public async Task<ActionResult<IEnumerable<ArticleDto>>> GetByAuthor(int authorId)
    {
        var articles = await _articleService.GetByAuthorAsync(authorId);
        return Ok(articles);
    }

    [HttpGet("featured")]
    [AllowAnonymous]
    public async Task<ActionResult<IEnumerable<ArticleDto>>> GetFeatured()
    {
        var articles = await _articleService.GetFeaturedAsync();
        return Ok(articles);
    }

    [HttpPost]
    public async Task<ActionResult<ArticleDto>> Create([FromBody] CreateArticleRequest request)
    {
        var userId = GetCurrentUserId();
        var article = await _articleService.CreateAsync(request, userId);
        return CreatedAtAction(nameof(GetById), new { id = article.Id }, article);
    }

    [HttpPut("{id:int}")]
    public async Task<ActionResult<ArticleDto>> Update(int id, [FromBody] UpdateArticleRequest request)
    {
        var userId = GetCurrentUserId();
        var article = await _articleService.UpdateAsync(id, request, userId);
        if (article == null)
        {
            return NotFound();
        }
        await _cacheService.InvalidatePatternAsync("articles:*");
        return Ok(article);
    }

    [HttpDelete("{id:int}")]
    public async Task<ActionResult> Delete(int id)
    {
        var userId = GetCurrentUserId();
        var result = await _articleService.DeleteAsync(id, userId);
        if (!result)
        {
            return NotFound();
        }
        return NoContent();
    }

    [HttpPost("{id:int}/publish")]
    public async Task<ActionResult<ArticleDto>> Publish(int id)
    {
        var article = await _articleService.PublishAsync(id);
        return Ok(article);
    }

    [HttpPost("{id:int}/unpublish")]
    public async Task<ActionResult<ArticleDto>> Unpublish(int id)
    {
        var article = await _articleService.UnpublishAsync(id);
        return Ok(article);
    }

    [HttpGet("{id:int}/comments")]
    [AllowAnonymous]
    public async Task<ActionResult<IEnumerable<CommentDto>>> GetComments(int id)
    {
        var comments = await _articleService.GetCommentsAsync(id);
        return Ok(comments);
    }

    [HttpPost("{id:int}/comments")]
    public async Task<ActionResult<CommentDto>> AddComment(int id, [FromBody] CreateCommentRequest request)
    {
        var userId = GetCurrentUserId();
        var comment = await _articleService.AddCommentAsync(id, request, userId);
        return Ok(comment);
    }

    [HttpPost("{id:int}/like")]
    public async Task<ActionResult> Like(int id)
    {
        var userId = GetCurrentUserId();
        await _articleService.LikeAsync(id, userId);
        return Ok();
    }

    private int GetCurrentUserId()
    {
        var claim = User.FindFirst("sub");
        return int.Parse(claim?.Value ?? "0");
    }
}
