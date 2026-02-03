using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace CSharpTestServer.Models;

[Table("Articles")]
public class Article
{
    public int Id { get; set; }

    [Required]
    [StringLength(200)]
    public string Title { get; set; } = "";

    [Required]
    public string Content { get; set; } = "";

    public string? Summary { get; set; }

    [StringLength(500)]
    public string? Slug { get; set; }

    public string? CoverImageUrl { get; set; }

    public ArticleStatus Status { get; set; } = ArticleStatus.Draft;

    public int AuthorId { get; set; }

    [ForeignKey("AuthorId")]
    public Person Author { get; set; } = null!;

    public int? CategoryId { get; set; }

    [ForeignKey("CategoryId")]
    public Category? Category { get; set; }

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    public DateTime? UpdatedAt { get; set; }

    public DateTime? PublishedAt { get; set; }

    public int ViewCount { get; set; }

    public int LikeCount { get; set; }

    public bool IsFeatured { get; set; }

    public ICollection<Comment> Comments { get; set; } = new List<Comment>();

    public ICollection<Tag> Tags { get; set; } = new List<Tag>();

    public ICollection<ArticleLike> Likes { get; set; } = new List<ArticleLike>();

    public void Publish()
    {
        Status = ArticleStatus.Published;
        PublishedAt = DateTime.UtcNow;
    }

    public void Unpublish()
    {
        Status = ArticleStatus.Draft;
        PublishedAt = null;
    }

    public void IncrementViewCount()
    {
        ViewCount++;
    }

    public void AddLike(int userId)
    {
        Likes.Add(new ArticleLike { ArticleId = Id, UserId = userId });
        LikeCount++;
    }

    public string GenerateSlug()
    {
        Slug = Title.ToLower()
            .Replace(" ", "-")
            .Replace(".", "")
            .Replace(",", "");
        return Slug;
    }
}

public enum ArticleStatus
{
    Draft,
    PendingReview,
    Published,
    Archived
}

public class Comment
{
    public int Id { get; set; }

    [Required]
    public string Content { get; set; } = "";

    public int ArticleId { get; set; }

    [ForeignKey("ArticleId")]
    public Article Article { get; set; } = null!;

    public int AuthorId { get; set; }

    [ForeignKey("AuthorId")]
    public Person Author { get; set; } = null!;

    public int? ParentCommentId { get; set; }

    [ForeignKey("ParentCommentId")]
    public Comment? ParentComment { get; set; }

    public ICollection<Comment> Replies { get; set; } = new List<Comment>();

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    public bool IsEdited { get; set; }
}

public class ArticleLike
{
    public int Id { get; set; }
    public int ArticleId { get; set; }
    public int UserId { get; set; }
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
}

public class Category
{
    public int Id { get; set; }

    [Required]
    [StringLength(100)]
    public string Name { get; set; } = "";

    public string? Description { get; set; }

    public string? Slug { get; set; }

    public int? ParentCategoryId { get; set; }

    [ForeignKey("ParentCategoryId")]
    public Category? ParentCategory { get; set; }

    public ICollection<Category> SubCategories { get; set; } = new List<Category>();

    public ICollection<Article> Articles { get; set; } = new List<Article>();
}

public class Tag
{
    public int Id { get; set; }

    [Required]
    [StringLength(50)]
    public string Name { get; set; } = "";

    public string? Slug { get; set; }

    public ICollection<Article> Articles { get; set; } = new List<Article>();
}
