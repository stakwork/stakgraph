using Microsoft.AspNetCore.Mvc;
using CSharpTestServer.DTOs;
using CSharpTestServer.Services;

namespace CSharpTestServer.Controllers;

[ApiController]
[Route("api/[controller]")]
public class ProductController : ControllerBase
{
    private readonly IProductService _productService;
    private readonly ICacheService _cacheService;
    private readonly IFileStorageService _fileStorageService;

    public ProductController(
        IProductService productService,
        ICacheService cacheService,
        IFileStorageService fileStorageService)
    {
        _productService = productService;
        _cacheService = cacheService;
        _fileStorageService = fileStorageService;
    }

    [HttpGet]
    public async Task<ActionResult<PagedResult<ProductDto>>> GetAll(
        [FromQuery] ProductQueryParams queryParams)
    {
        var products = await _productService.GetAllAsync(queryParams);
        return Ok(products);
    }

    [HttpGet("{id:int}")]
    public async Task<ActionResult<ProductDto>> GetById(int id)
    {
        var product = await _productService.GetByIdAsync(id);
        if (product == null)
        {
            return NotFound();
        }
        return Ok(product);
    }

    [HttpGet("category/{categoryId:int}")]
    public async Task<ActionResult<IEnumerable<ProductDto>>> GetByCategory(int categoryId)
    {
        var products = await _productService.GetByCategoryAsync(categoryId);
        return Ok(products);
    }

    [HttpGet("search")]
    public async Task<ActionResult<IEnumerable<ProductDto>>> Search(
        [FromQuery] string q,
        [FromQuery] decimal? minPrice,
        [FromQuery] decimal? maxPrice)
    {
        var products = await _productService.SearchAsync(q, minPrice, maxPrice);
        return Ok(products);
    }

    [HttpPost]
    public async Task<ActionResult<ProductDto>> Create([FromBody] CreateProductRequest request)
    {
        var product = await _productService.CreateAsync(request);
        return CreatedAtAction(nameof(GetById), new { id = product.Id }, product);
    }

    [HttpPut("{id:int}")]
    public async Task<ActionResult<ProductDto>> Update(int id, [FromBody] UpdateProductRequest request)
    {
        var product = await _productService.UpdateAsync(id, request);
        await _cacheService.InvalidateAsync($"product:{id}");
        return Ok(product);
    }

    [HttpDelete("{id:int}")]
    public async Task<ActionResult> Delete(int id)
    {
        await _productService.DeleteAsync(id);
        return NoContent();
    }

    [HttpPost("{id:int}/images")]
    public async Task<ActionResult<string>> UploadImage(int id, IFormFile file)
    {
        var url = await _fileStorageService.UploadAsync(file, $"products/{id}");
        await _productService.AddImageAsync(id, url);
        return Ok(new { Url = url });
    }

    [HttpDelete("{id:int}/images/{imageId:int}")]
    public async Task<ActionResult> DeleteImage(int id, int imageId)
    {
        await _productService.RemoveImageAsync(id, imageId);
        return NoContent();
    }

    [HttpGet("{id:int}/reviews")]
    public async Task<ActionResult<IEnumerable<ReviewDto>>> GetReviews(int id)
    {
        var reviews = await _productService.GetReviewsAsync(id);
        return Ok(reviews);
    }

    [HttpPost("{id:int}/reviews")]
    public async Task<ActionResult<ReviewDto>> AddReview(int id, [FromBody] CreateReviewRequest request)
    {
        var review = await _productService.AddReviewAsync(id, request);
        return Ok(review);
    }

    [HttpPatch("{id:int}/inventory")]
    public async Task<ActionResult<ProductDto>> UpdateInventory(int id, [FromBody] UpdateInventoryRequest request)
    {
        var product = await _productService.UpdateInventoryAsync(id, request.Quantity);
        return Ok(product);
    }
}
