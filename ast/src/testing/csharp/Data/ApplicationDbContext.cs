using Microsoft.EntityFrameworkCore;
using CSharpTestServer.Models;

namespace CSharpTestServer.Data;

public class ApplicationDbContext : DbContext
{
    public ApplicationDbContext(DbContextOptions<ApplicationDbContext> options)
        : base(options)
    {
    }

    public DbSet<Person> Persons { get; set; }
    public DbSet<UserProfile> UserProfiles { get; set; }
    public DbSet<Article> Articles { get; set; }
    public DbSet<Comment> Comments { get; set; }
    public DbSet<ArticleLike> ArticleLikes { get; set; }
    public DbSet<Category> Categories { get; set; }
    public DbSet<Tag> Tags { get; set; }
    public DbSet<Order> Orders { get; set; }
    public DbSet<OrderItem> OrderItems { get; set; }
    public DbSet<Product> Products { get; set; }
    public DbSet<ProductImage> ProductImages { get; set; }
    public DbSet<ProductReview> ProductReviews { get; set; }
    public DbSet<ProductVariant> ProductVariants { get; set; }
    public DbSet<Brand> Brands { get; set; }
    public DbSet<Country> Countries { get; set; }

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        base.OnModelCreating(modelBuilder);

        modelBuilder.Entity<Person>(entity =>
        {
            entity.HasIndex(e => e.Email).IsUnique();
            entity.HasOne(e => e.Country)
                .WithMany(c => c.Persons)
                .HasForeignKey(e => e.CountryId)
                .OnDelete(DeleteBehavior.SetNull);
        });

        modelBuilder.Entity<UserProfile>(entity =>
        {
            entity.HasOne(e => e.Person)
                .WithOne(p => p.Profile)
                .HasForeignKey<UserProfile>(e => e.PersonId);
            entity.OwnsOne(e => e.Preferences);
        });

        modelBuilder.Entity<Article>(entity =>
        {
            entity.HasIndex(e => e.Slug).IsUnique();
            entity.HasOne(e => e.Author)
                .WithMany(p => p.Articles)
                .HasForeignKey(e => e.AuthorId)
                .OnDelete(DeleteBehavior.Cascade);
            entity.HasMany(e => e.Tags)
                .WithMany(t => t.Articles);
        });

        modelBuilder.Entity<Comment>(entity =>
        {
            entity.HasOne(e => e.Article)
                .WithMany(a => a.Comments)
                .HasForeignKey(e => e.ArticleId)
                .OnDelete(DeleteBehavior.Cascade);
            entity.HasOne(e => e.ParentComment)
                .WithMany(c => c.Replies)
                .HasForeignKey(e => e.ParentCommentId)
                .OnDelete(DeleteBehavior.Restrict);
        });

        modelBuilder.Entity<Order>(entity =>
        {
            entity.HasIndex(e => e.OrderNumber).IsUnique();
            entity.HasOne(e => e.Customer)
                .WithMany(p => p.Orders)
                .HasForeignKey(e => e.CustomerId)
                .OnDelete(DeleteBehavior.Restrict);
            entity.OwnsOne(e => e.ShippingAddress);
            entity.OwnsOne(e => e.BillingAddress);
            entity.OwnsOne(e => e.Tracking, tracking =>
            {
                tracking.OwnsMany(t => t.Events);
            });
        });

        modelBuilder.Entity<OrderItem>(entity =>
        {
            entity.HasOne(e => e.Order)
                .WithMany(o => o.Items)
                .HasForeignKey(e => e.OrderId)
                .OnDelete(DeleteBehavior.Cascade);
        });

        modelBuilder.Entity<Product>(entity =>
        {
            entity.HasIndex(e => e.Sku).IsUnique();
            entity.HasOne(e => e.Category)
                .WithMany(c => c.Articles)
                .HasForeignKey(e => e.CategoryId)
                .OnDelete(DeleteBehavior.Restrict);
        });

        modelBuilder.Entity<ProductVariant>(entity =>
        {
            entity.Property(e => e.Attributes)
                .HasConversion(
                    v => System.Text.Json.JsonSerializer.Serialize(v, (System.Text.Json.JsonSerializerOptions?)null),
                    v => System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, string>>(v, (System.Text.Json.JsonSerializerOptions?)null) ?? new());
        });

        modelBuilder.Entity<Category>(entity =>
        {
            entity.HasOne(e => e.ParentCategory)
                .WithMany(c => c.SubCategories)
                .HasForeignKey(e => e.ParentCategoryId)
                .OnDelete(DeleteBehavior.Restrict);
        });

        SeedData(modelBuilder);
    }

    private void SeedData(ModelBuilder modelBuilder)
    {
        modelBuilder.Entity<Country>().HasData(
            new Country { Id = 1, Name = "United States", Code = "US" },
            new Country { Id = 2, Name = "United Kingdom", Code = "GB" },
            new Country { Id = 3, Name = "Canada", Code = "CA" },
            new Country { Id = 4, Name = "Australia", Code = "AU" },
            new Country { Id = 5, Name = "Germany", Code = "DE" }
        );

        modelBuilder.Entity<Category>().HasData(
            new Category { Id = 1, Name = "Electronics", Slug = "electronics" },
            new Category { Id = 2, Name = "Clothing", Slug = "clothing" },
            new Category { Id = 3, Name = "Books", Slug = "books" },
            new Category { Id = 4, Name = "Home & Garden", Slug = "home-garden" }
        );

        modelBuilder.Entity<Brand>().HasData(
            new Brand { Id = 1, Name = "TechCorp", Website = "https://techcorp.example.com" },
            new Brand { Id = 2, Name = "StyleBrand", Website = "https://stylebrand.example.com" },
            new Brand { Id = 3, Name = "BookHouse", Website = "https://bookhouse.example.com" }
        );
    }
}
