using CSharpTestServer.DTOs;
using CSharpTestServer.Models;
using CSharpTestServer.Repositories;
using AutoMapper;

namespace CSharpTestServer.Services;

public interface IArticleService
{
    Task<PagedResult<ArticleDto>> GetAllPagedAsync(int page, int pageSize, string? sortBy);
    Task<ArticleDto?> GetByIdAsync(int id);
    Task<IEnumerable<ArticleDto>> GetByAuthorAsync(int authorId);
    Task<IEnumerable<ArticleDto>> GetFeaturedAsync();
    Task<ArticleDto> CreateAsync(CreateArticleRequest request, int authorId);
    Task<ArticleDto?> UpdateAsync(int id, UpdateArticleRequest request, int userId);
    Task<bool> DeleteAsync(int id, int userId);
    Task<ArticleDto> PublishAsync(int id);
    Task<ArticleDto> UnpublishAsync(int id);
    Task<IEnumerable<CommentDto>> GetCommentsAsync(int articleId);
    Task<CommentDto> AddCommentAsync(int articleId, CreateCommentRequest request, int userId);
    Task LikeAsync(int articleId, int userId);
}

public class ArticleService : IArticleService
{
    private readonly IArticleRepository _articleRepository;
    private readonly IMapper _mapper;
    private readonly ILogger<ArticleService> _logger;

    public ArticleService(
        IArticleRepository articleRepository,
        IMapper mapper,
        ILogger<ArticleService> logger)
    {
        _articleRepository = articleRepository;
        _mapper = mapper;
        _logger = logger;
    }

    public async Task<PagedResult<ArticleDto>> GetAllPagedAsync(int page, int pageSize, string? sortBy)
    {
        var (articles, totalCount) = await _articleRepository.GetPagedAsync(page, pageSize, sortBy);
        var items = _mapper.Map<IEnumerable<ArticleDto>>(articles);
        
        return new PagedResult<ArticleDto>
        {
            Items = items,
            Page = page,
            PageSize = pageSize,
            TotalCount = totalCount,
            TotalPages = (int)Math.Ceiling(totalCount / (double)pageSize)
        };
    }

    public async Task<ArticleDto?> GetByIdAsync(int id)
    {
        var article = await _articleRepository.GetByIdWithDetailsAsync(id);
        if (article == null)
        {
            return null;
        }
        
        article.IncrementViewCount();
        await _articleRepository.UpdateAsync(article);
        
        return _mapper.Map<ArticleDto>(article);
    }

    public async Task<IEnumerable<ArticleDto>> GetByAuthorAsync(int authorId)
    {
        var articles = await _articleRepository.GetByAuthorIdAsync(authorId);
        return _mapper.Map<IEnumerable<ArticleDto>>(articles);
    }

    public async Task<IEnumerable<ArticleDto>> GetFeaturedAsync()
    {
        var articles = await _articleRepository.GetFeaturedAsync();
        return _mapper.Map<IEnumerable<ArticleDto>>(articles);
    }

    public async Task<ArticleDto> CreateAsync(CreateArticleRequest request, int authorId)
    {
        var article = new Article
        {
            Title = request.Title,
            Content = request.Content,
            Summary = request.Summary,
            AuthorId = authorId,
            CategoryId = request.CategoryId
        };
        
        article.GenerateSlug();
        await _articleRepository.AddAsync(article);
        
        _logger.LogInformation("Article {Id} created by user {AuthorId}", article.Id, authorId);
        return _mapper.Map<ArticleDto>(article);
    }

    public async Task<ArticleDto?> UpdateAsync(int id, UpdateArticleRequest request, int userId)
    {
        var article = await _articleRepository.GetByIdAsync(id);
        if (article == null || article.AuthorId != userId)
        {
            return null;
        }

        article.Title = request.Title;
        article.Content = request.Content;
        article.Summary = request.Summary;
        article.CategoryId = request.CategoryId;
        article.UpdatedAt = DateTime.UtcNow;
        article.GenerateSlug();

        await _articleRepository.UpdateAsync(article);
        return _mapper.Map<ArticleDto>(article);
    }

    public async Task<bool> DeleteAsync(int id, int userId)
    {
        var article = await _articleRepository.GetByIdAsync(id);
        if (article == null || article.AuthorId != userId)
        {
            return false;
        }

        await _articleRepository.DeleteAsync(id);
        return true;
    }

    public async Task<ArticleDto> PublishAsync(int id)
    {
        var article = await _articleRepository.GetByIdAsync(id);
        if (article == null)
        {
            throw new KeyNotFoundException($"Article {id} not found");
        }

        article.Publish();
        await _articleRepository.UpdateAsync(article);
        
        _logger.LogInformation("Article {Id} published", id);
        return _mapper.Map<ArticleDto>(article);
    }

    public async Task<ArticleDto> UnpublishAsync(int id)
    {
        var article = await _articleRepository.GetByIdAsync(id);
        if (article == null)
        {
            throw new KeyNotFoundException($"Article {id} not found");
        }

        article.Unpublish();
        await _articleRepository.UpdateAsync(article);
        
        return _mapper.Map<ArticleDto>(article);
    }

    public async Task<IEnumerable<CommentDto>> GetCommentsAsync(int articleId)
    {
        var comments = await _articleRepository.GetCommentsAsync(articleId);
        return _mapper.Map<IEnumerable<CommentDto>>(comments);
    }

    public async Task<CommentDto> AddCommentAsync(int articleId, CreateCommentRequest request, int userId)
    {
        var comment = new Comment
        {
            ArticleId = articleId,
            Content = request.Content,
            AuthorId = userId,
            ParentCommentId = request.ParentCommentId
        };

        await _articleRepository.AddCommentAsync(comment);
        return _mapper.Map<CommentDto>(comment);
    }

    public async Task LikeAsync(int articleId, int userId)
    {
        var article = await _articleRepository.GetByIdAsync(articleId);
        if (article == null)
        {
            throw new KeyNotFoundException($"Article {articleId} not found");
        }

        article.AddLike(userId);
        await _articleRepository.UpdateAsync(article);
    }
}
