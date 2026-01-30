using Microsoft.EntityFrameworkCore;
using CSharpTestServer.Data;
using CSharpTestServer.Models;

namespace CSharpTestServer.Repositories;

public interface IRepository<T> where T : class
{
    Task<T?> GetByIdAsync(int id);
    Task<IEnumerable<T>> GetAllAsync();
    Task AddAsync(T entity);
    Task UpdateAsync(T entity);
    Task DeleteAsync(int id);
}

public interface IPersonRepository : IRepository<Person>
{
    Task<IEnumerable<Person>> GetAllAsync(int page, int pageSize);
    Task<IEnumerable<Person>> SearchAsync(string query);
    Task<Person?> GetByEmailAsync(string email);
}

public class PersonRepository : IPersonRepository
{
    private readonly ApplicationDbContext _context;

    public PersonRepository(ApplicationDbContext context)
    {
        _context = context;
    }

    public async Task<Person?> GetByIdAsync(int id)
    {
        return await _context.Persons
            .Include(p => p.Country)
            .Include(p => p.Profile)
            .FirstOrDefaultAsync(p => p.Id == id);
    }

    public async Task<IEnumerable<Person>> GetAllAsync()
    {
        return await _context.Persons
            .Include(p => p.Country)
            .ToListAsync();
    }

    public async Task<IEnumerable<Person>> GetAllAsync(int page, int pageSize)
    {
        return await _context.Persons
            .Include(p => p.Country)
            .OrderBy(p => p.LastName)
            .ThenBy(p => p.FirstName)
            .Skip((page - 1) * pageSize)
            .Take(pageSize)
            .ToListAsync();
    }

    public async Task<IEnumerable<Person>> SearchAsync(string query)
    {
        var lowerQuery = query.ToLower();
        return await _context.Persons
            .Where(p => p.FirstName.ToLower().Contains(lowerQuery) ||
                        p.LastName.ToLower().Contains(lowerQuery) ||
                        p.Email.ToLower().Contains(lowerQuery))
            .ToListAsync();
    }

    public async Task<Person?> GetByEmailAsync(string email)
    {
        return await _context.Persons
            .FirstOrDefaultAsync(p => p.Email == email);
    }

    public async Task AddAsync(Person entity)
    {
        await _context.Persons.AddAsync(entity);
        await _context.SaveChangesAsync();
    }

    public async Task UpdateAsync(Person entity)
    {
        _context.Persons.Update(entity);
        await _context.SaveChangesAsync();
    }

    public async Task DeleteAsync(int id)
    {
        var entity = await GetByIdAsync(id);
        if (entity != null)
        {
            _context.Persons.Remove(entity);
            await _context.SaveChangesAsync();
        }
    }
}

public interface IArticleRepository : IRepository<Article>
{
    Task<(IEnumerable<Article>, int)> GetPagedAsync(int page, int pageSize, string? sortBy);
    Task<Article?> GetByIdWithDetailsAsync(int id);
    Task<IEnumerable<Article>> GetByAuthorIdAsync(int authorId);
    Task<IEnumerable<Article>> GetFeaturedAsync();
    Task<IEnumerable<Comment>> GetCommentsAsync(int articleId);
    Task AddCommentAsync(Comment comment);
}

public class ArticleRepository : IArticleRepository
{
    private readonly ApplicationDbContext _context;

    public ArticleRepository(ApplicationDbContext context)
    {
        _context = context;
    }

    public async Task<Article?> GetByIdAsync(int id)
    {
        return await _context.Articles.FindAsync(id);
    }

    public async Task<Article?> GetByIdWithDetailsAsync(int id)
    {
        return await _context.Articles
            .Include(a => a.Author)
            .Include(a => a.Category)
            .Include(a => a.Tags)
            .Include(a => a.Comments)
            .FirstOrDefaultAsync(a => a.Id == id);
    }

    public async Task<IEnumerable<Article>> GetAllAsync()
    {
        return await _context.Articles
            .Include(a => a.Author)
            .ToListAsync();
    }

    public async Task<(IEnumerable<Article>, int)> GetPagedAsync(int page, int pageSize, string? sortBy)
    {
        var query = _context.Articles
            .Include(a => a.Author)
            .Include(a => a.Category)
            .Where(a => a.Status == ArticleStatus.Published);

        query = sortBy?.ToLower() switch
        {
            "created" => query.OrderByDescending(a => a.CreatedAt),
            "views" => query.OrderByDescending(a => a.ViewCount),
            "likes" => query.OrderByDescending(a => a.LikeCount),
            _ => query.OrderByDescending(a => a.PublishedAt)
        };

        var totalCount = await query.CountAsync();
        var items = await query
            .Skip((page - 1) * pageSize)
            .Take(pageSize)
            .ToListAsync();

        return (items, totalCount);
    }

    public async Task<IEnumerable<Article>> GetByAuthorIdAsync(int authorId)
    {
        return await _context.Articles
            .Where(a => a.AuthorId == authorId)
            .OrderByDescending(a => a.CreatedAt)
            .ToListAsync();
    }

    public async Task<IEnumerable<Article>> GetFeaturedAsync()
    {
        return await _context.Articles
            .Where(a => a.IsFeatured && a.Status == ArticleStatus.Published)
            .OrderByDescending(a => a.PublishedAt)
            .Take(5)
            .ToListAsync();
    }

    public async Task<IEnumerable<Comment>> GetCommentsAsync(int articleId)
    {
        return await _context.Comments
            .Include(c => c.Author)
            .Include(c => c.Replies)
            .Where(c => c.ArticleId == articleId && c.ParentCommentId == null)
            .OrderByDescending(c => c.CreatedAt)
            .ToListAsync();
    }

    public async Task AddCommentAsync(Comment comment)
    {
        await _context.Comments.AddAsync(comment);
        await _context.SaveChangesAsync();
    }

    public async Task AddAsync(Article entity)
    {
        await _context.Articles.AddAsync(entity);
        await _context.SaveChangesAsync();
    }

    public async Task UpdateAsync(Article entity)
    {
        _context.Articles.Update(entity);
        await _context.SaveChangesAsync();
    }

    public async Task DeleteAsync(int id)
    {
        var entity = await GetByIdAsync(id);
        if (entity != null)
        {
            _context.Articles.Remove(entity);
            await _context.SaveChangesAsync();
        }
    }
}

public interface IOrderRepository
{
    Task<Order?> GetByIdAsync(Guid id);
    Task<Order?> GetByIdWithItemsAsync(Guid id);
    Task<(IEnumerable<Order>, int)> GetAllAsync(DTOs.OrderQueryParams queryParams);
    Task<IEnumerable<Order>> GetByUserIdAsync(int userId);
    Task AddAsync(Order entity);
    Task UpdateAsync(Order entity);
}

public class OrderRepository : IOrderRepository
{
    private readonly ApplicationDbContext _context;

    public OrderRepository(ApplicationDbContext context)
    {
        _context = context;
    }

    public async Task<Order?> GetByIdAsync(Guid id)
    {
        return await _context.Orders.FindAsync(id);
    }

    public async Task<Order?> GetByIdWithItemsAsync(Guid id)
    {
        return await _context.Orders
            .Include(o => o.Items)
            .ThenInclude(i => i.Product)
            .Include(o => o.Customer)
            .FirstOrDefaultAsync(o => o.Id == id);
    }

    public async Task<(IEnumerable<Order>, int)> GetAllAsync(DTOs.OrderQueryParams queryParams)
    {
        var query = _context.Orders
            .Include(o => o.Customer)
            .Include(o => o.Items)
            .AsQueryable();

        if (queryParams.Status.HasValue)
        {
            query = query.Where(o => o.Status == queryParams.Status.Value);
        }

        if (queryParams.FromDate.HasValue)
        {
            query = query.Where(o => o.CreatedAt >= queryParams.FromDate.Value);
        }

        if (queryParams.ToDate.HasValue)
        {
            query = query.Where(o => o.CreatedAt <= queryParams.ToDate.Value);
        }

        if (!string.IsNullOrEmpty(queryParams.CustomerEmail))
        {
            query = query.Where(o => o.CustomerEmail.Contains(queryParams.CustomerEmail));
        }

        var totalCount = await query.CountAsync();
        var items = await query
            .OrderByDescending(o => o.CreatedAt)
            .Skip((queryParams.Page - 1) * queryParams.PageSize)
            .Take(queryParams.PageSize)
            .ToListAsync();

        return (items, totalCount);
    }

    public async Task<IEnumerable<Order>> GetByUserIdAsync(int userId)
    {
        return await _context.Orders
            .Include(o => o.Items)
            .Where(o => o.CustomerId == userId)
            .OrderByDescending(o => o.CreatedAt)
            .ToListAsync();
    }

    public async Task AddAsync(Order entity)
    {
        await _context.Orders.AddAsync(entity);
        await _context.SaveChangesAsync();
    }

    public async Task UpdateAsync(Order entity)
    {
        _context.Orders.Update(entity);
        await _context.SaveChangesAsync();
    }
}

public interface IProductRepository : IRepository<Product>
{
    Task<(IEnumerable<Product>, int)> GetAllAsync(DTOs.ProductQueryParams queryParams);
    Task<IEnumerable<Product>> GetByCategoryIdAsync(int categoryId);
    Task<IEnumerable<Product>> SearchAsync(string query, decimal? minPrice, decimal? maxPrice);
}

public class ProductRepository : IProductRepository
{
    private readonly ApplicationDbContext _context;

    public ProductRepository(ApplicationDbContext context)
    {
        _context = context;
    }

    public async Task<Product?> GetByIdAsync(int id)
    {
        return await _context.Products
            .Include(p => p.Category)
            .Include(p => p.Brand)
            .Include(p => p.Images)
            .Include(p => p.Reviews)
            .FirstOrDefaultAsync(p => p.Id == id);
    }

    public async Task<IEnumerable<Product>> GetAllAsync()
    {
        return await _context.Products
            .Include(p => p.Category)
            .ToListAsync();
    }

    public async Task<(IEnumerable<Product>, int)> GetAllAsync(DTOs.ProductQueryParams queryParams)
    {
        var query = _context.Products
            .Include(p => p.Category)
            .Include(p => p.Brand)
            .Include(p => p.Images)
            .Where(p => p.IsActive);

        if (queryParams.CategoryId.HasValue)
        {
            query = query.Where(p => p.CategoryId == queryParams.CategoryId.Value);
        }

        if (queryParams.BrandId.HasValue)
        {
            query = query.Where(p => p.BrandId == queryParams.BrandId.Value);
        }

        if (queryParams.MinPrice.HasValue)
        {
            query = query.Where(p => p.Price >= queryParams.MinPrice.Value);
        }

        if (queryParams.MaxPrice.HasValue)
        {
            query = query.Where(p => p.Price <= queryParams.MaxPrice.Value);
        }

        if (queryParams.InStock.HasValue && queryParams.InStock.Value)
        {
            query = query.Where(p => p.StockQuantity > 0);
        }

        query = queryParams.SortBy?.ToLower() switch
        {
            "price" => queryParams.SortOrder == "desc"
                ? query.OrderByDescending(p => p.Price)
                : query.OrderBy(p => p.Price),
            "name" => query.OrderBy(p => p.Name),
            "created" => query.OrderByDescending(p => p.CreatedAt),
            _ => query.OrderByDescending(p => p.CreatedAt)
        };

        var totalCount = await query.CountAsync();
        var items = await query
            .Skip((queryParams.Page - 1) * queryParams.PageSize)
            .Take(queryParams.PageSize)
            .ToListAsync();

        return (items, totalCount);
    }

    public async Task<IEnumerable<Product>> GetByCategoryIdAsync(int categoryId)
    {
        return await _context.Products
            .Where(p => p.CategoryId == categoryId && p.IsActive)
            .ToListAsync();
    }

    public async Task<IEnumerable<Product>> SearchAsync(string query, decimal? minPrice, decimal? maxPrice)
    {
        var lowerQuery = query.ToLower();
        var q = _context.Products
            .Include(p => p.Category)
            .Where(p => p.IsActive &&
                       (p.Name.ToLower().Contains(lowerQuery) ||
                        (p.Description != null && p.Description.ToLower().Contains(lowerQuery))));

        if (minPrice.HasValue)
        {
            q = q.Where(p => p.Price >= minPrice.Value);
        }

        if (maxPrice.HasValue)
        {
            q = q.Where(p => p.Price <= maxPrice.Value);
        }

        return await q.ToListAsync();
    }

    public async Task AddAsync(Product entity)
    {
        await _context.Products.AddAsync(entity);
        await _context.SaveChangesAsync();
    }

    public async Task UpdateAsync(Product entity)
    {
        _context.Products.Update(entity);
        await _context.SaveChangesAsync();
    }

    public async Task DeleteAsync(int id)
    {
        var entity = await GetByIdAsync(id);
        if (entity != null)
        {
            _context.Products.Remove(entity);
            await _context.SaveChangesAsync();
        }
    }
}
