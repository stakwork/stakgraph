using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace CSharpTestServer.Models;

[Table("Products")]
public class Product
{
    public int Id { get; set; }

    [Required]
    [StringLength(200)]
    public string Name { get; set; } = "";

    public string? Description { get; set; }

    [StringLength(100)]
    public string? Sku { get; set; }

    [Column(TypeName = "decimal(18,2)")]
    public decimal Price { get; set; }

    [Column(TypeName = "decimal(18,2)")]
    public decimal? CompareAtPrice { get; set; }

    public int StockQuantity { get; set; }

    public int LowStockThreshold { get; set; } = 10;

    public bool IsActive { get; set; } = true;

    public bool IsFeatured { get; set; }

    public int CategoryId { get; set; }

    [ForeignKey("CategoryId")]
    public Category Category { get; set; } = null!;

    public int? BrandId { get; set; }

    [ForeignKey("BrandId")]
    public Brand? Brand { get; set; }

    public decimal Weight { get; set; }

    public string? Dimensions { get; set; }

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    public DateTime? UpdatedAt { get; set; }

    public ICollection<ProductImage> Images { get; set; } = new List<ProductImage>();

    public ICollection<ProductReview> Reviews { get; set; } = new List<ProductReview>();

    public ICollection<ProductVariant> Variants { get; set; } = new List<ProductVariant>();

    public double AverageRating => Reviews.Any() ? Reviews.Average(r => r.Rating) : 0;

    public bool IsInStock => StockQuantity > 0;

    public bool IsLowStock => StockQuantity <= LowStockThreshold;

    public decimal DiscountPercentage
    {
        get
        {
            if (CompareAtPrice.HasValue && CompareAtPrice > 0)
            {
                return Math.Round((1 - (Price / CompareAtPrice.Value)) * 100, 2);
            }
            return 0;
        }
    }

    public void UpdateStock(int quantity)
    {
        StockQuantity = quantity;
        UpdatedAt = DateTime.UtcNow;
    }

    public void DecrementStock(int quantity)
    {
        if (StockQuantity < quantity)
        {
            throw new InvalidOperationException("Insufficient stock");
        }
        StockQuantity -= quantity;
        UpdatedAt = DateTime.UtcNow;
    }

    public void IncrementStock(int quantity)
    {
        StockQuantity += quantity;
        UpdatedAt = DateTime.UtcNow;
    }
}

public class ProductImage
{
    public int Id { get; set; }

    public int ProductId { get; set; }

    [ForeignKey("ProductId")]
    public Product Product { get; set; } = null!;

    [Required]
    public string Url { get; set; } = "";

    public string? AltText { get; set; }

    public int SortOrder { get; set; }

    public bool IsPrimary { get; set; }
}

public class ProductReview
{
    public int Id { get; set; }

    public int ProductId { get; set; }

    [ForeignKey("ProductId")]
    public Product Product { get; set; } = null!;

    public int UserId { get; set; }

    [ForeignKey("UserId")]
    public Person User { get; set; } = null!;

    [Range(1, 5)]
    public int Rating { get; set; }

    [StringLength(500)]
    public string? Title { get; set; }

    public string? Content { get; set; }

    public bool IsVerifiedPurchase { get; set; }

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    public int HelpfulCount { get; set; }
}

public class ProductVariant
{
    public int Id { get; set; }

    public int ProductId { get; set; }

    [ForeignKey("ProductId")]
    public Product Product { get; set; } = null!;

    [StringLength(100)]
    public string Name { get; set; } = "";

    [StringLength(100)]
    public string? Sku { get; set; }

    [Column(TypeName = "decimal(18,2)")]
    public decimal? PriceAdjustment { get; set; }

    public int StockQuantity { get; set; }

    public Dictionary<string, string> Attributes { get; set; } = new();
}

public class Brand
{
    public int Id { get; set; }

    [Required]
    [StringLength(100)]
    public string Name { get; set; } = "";

    public string? Description { get; set; }

    public string? LogoUrl { get; set; }

    public string? Website { get; set; }

    public ICollection<Product> Products { get; set; } = new List<Product>();
}

public class Country
{
    public int Id { get; set; }

    [Required]
    [StringLength(100)]
    public string Name { get; set; } = "";

    [StringLength(2)]
    public string Code { get; set; } = "";

    public ICollection<Person> Persons { get; set; } = new List<Person>();
}
