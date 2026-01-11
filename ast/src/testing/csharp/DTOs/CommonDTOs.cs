using System.ComponentModel.DataAnnotations;
using CSharpTestServer.Models;

namespace CSharpTestServer.DTOs;

public class PersonDto
{
    public int Id { get; set; }
    public string FirstName { get; set; } = "";
    public string LastName { get; set; } = "";
    public string FullName { get; set; } = "";
    public string Email { get; set; } = "";
    public string? Phone { get; set; }
    public DateTime DateOfBirth { get; set; }
    public int Age { get; set; }
    public string? AvatarUrl { get; set; }
    public bool IsActive { get; set; }
    public DateTime CreatedAt { get; set; }
    public string? Country { get; set; }
}

public class CreatePersonRequest
{
    [Required]
    [StringLength(100)]
    public string FirstName { get; set; } = "";

    [Required]
    [StringLength(100)]
    public string LastName { get; set; } = "";

    [Required]
    [EmailAddress]
    public string Email { get; set; } = "";

    [Phone]
    public string? Phone { get; set; }

    [Required]
    public DateTime DateOfBirth { get; set; }

    public int? CountryId { get; set; }
}

public class UpdatePersonRequest
{
    [Required]
    [StringLength(100)]
    public string FirstName { get; set; } = "";

    [Required]
    [StringLength(100)]
    public string LastName { get; set; } = "";

    [Required]
    [EmailAddress]
    public string Email { get; set; } = "";

    public string? Phone { get; set; }
    public DateTime DateOfBirth { get; set; }
    public int? CountryId { get; set; }
}

public class PatchPersonRequest
{
    public string? FirstName { get; set; }
    public string? LastName { get; set; }
    public string? Email { get; set; }
    public string? Phone { get; set; }
}

public class ArticleDto
{
    public int Id { get; set; }
    public string Title { get; set; } = "";
    public string Content { get; set; } = "";
    public string? Summary { get; set; }
    public string? Slug { get; set; }
    public string? CoverImageUrl { get; set; }
    public string Status { get; set; } = "";
    public string AuthorName { get; set; } = "";
    public int AuthorId { get; set; }
    public string? CategoryName { get; set; }
    public DateTime CreatedAt { get; set; }
    public DateTime? PublishedAt { get; set; }
    public int ViewCount { get; set; }
    public int LikeCount { get; set; }
    public int CommentCount { get; set; }
    public bool IsFeatured { get; set; }
    public IEnumerable<string> Tags { get; set; } = Array.Empty<string>();
}

public class CreateArticleRequest
{
    [Required]
    [StringLength(200)]
    public string Title { get; set; } = "";

    [Required]
    public string Content { get; set; } = "";

    public string? Summary { get; set; }
    public int? CategoryId { get; set; }
    public IEnumerable<int>? TagIds { get; set; }
}

public class UpdateArticleRequest
{
    [Required]
    [StringLength(200)]
    public string Title { get; set; } = "";

    [Required]
    public string Content { get; set; } = "";

    public string? Summary { get; set; }
    public int? CategoryId { get; set; }
    public IEnumerable<int>? TagIds { get; set; }
}

public class CommentDto
{
    public int Id { get; set; }
    public string Content { get; set; } = "";
    public string AuthorName { get; set; } = "";
    public int AuthorId { get; set; }
    public DateTime CreatedAt { get; set; }
    public bool IsEdited { get; set; }
    public int? ParentCommentId { get; set; }
    public IEnumerable<CommentDto> Replies { get; set; } = Array.Empty<CommentDto>();
}

public class CreateCommentRequest
{
    [Required]
    public string Content { get; set; } = "";

    public int? ParentCommentId { get; set; }
}

public class OrderDto
{
    public Guid Id { get; set; }
    public string OrderNumber { get; set; } = "";
    public string CustomerEmail { get; set; } = "";
    public string Status { get; set; } = "";
    public decimal SubTotal { get; set; }
    public decimal TaxAmount { get; set; }
    public decimal ShippingCost { get; set; }
    public decimal DiscountAmount { get; set; }
    public decimal TotalAmount { get; set; }
    public string PaymentMethod { get; set; } = "";
    public DateTime CreatedAt { get; set; }
    public DateTime? PaidAt { get; set; }
    public DateTime? ShippedAt { get; set; }
    public DateTime? DeliveredAt { get; set; }
    public AddressDto ShippingAddress { get; set; } = new();
    public IEnumerable<OrderItemDto> Items { get; set; } = Array.Empty<OrderItemDto>();
}

public class CreateOrderRequest
{
    [Required]
    public List<CreateOrderItemRequest> Items { get; set; } = new();

    [Required]
    public AddressDto ShippingAddress { get; set; } = new();

    public AddressDto? BillingAddress { get; set; }

    [Required]
    public PaymentMethod PaymentMethod { get; set; }

    public string? CouponCode { get; set; }
}

public class CreateOrderItemRequest
{
    public int ProductId { get; set; }
    public int Quantity { get; set; }
}

public class AddOrderItemRequest
{
    public int ProductId { get; set; }
    public int Quantity { get; set; }
}

public class OrderItemDto
{
    public int Id { get; set; }
    public int ProductId { get; set; }
    public string ProductName { get; set; } = "";
    public string? ProductSku { get; set; }
    public int Quantity { get; set; }
    public decimal UnitPrice { get; set; }
    public decimal TotalPrice { get; set; }
}

public class AddressDto
{
    public string Street { get; set; } = "";
    public string City { get; set; } = "";
    public string State { get; set; } = "";
    public string PostalCode { get; set; } = "";
    public string Country { get; set; } = "";
}

public class CheckoutRequest
{
    [Required]
    public PaymentMethod PaymentMethod { get; set; }

    public string? CardToken { get; set; }
}

public class CancelOrderRequest
{
    [Required]
    public string Reason { get; set; } = "";
}

public class UpdateOrderStatusRequest
{
    [Required]
    public OrderStatus Status { get; set; }
}

public class TrackingInfoDto
{
    public string TrackingNumber { get; set; } = "";
    public string Carrier { get; set; } = "";
    public DateTime? EstimatedDelivery { get; set; }
    public IEnumerable<TrackingEventDto> Events { get; set; } = Array.Empty<TrackingEventDto>();
}

public class TrackingEventDto
{
    public DateTime Timestamp { get; set; }
    public string Status { get; set; } = "";
    public string Location { get; set; } = "";
    public string? Description { get; set; }
}

public class ProductDto
{
    public int Id { get; set; }
    public string Name { get; set; } = "";
    public string? Description { get; set; }
    public string? Sku { get; set; }
    public decimal Price { get; set; }
    public decimal? CompareAtPrice { get; set; }
    public decimal DiscountPercentage { get; set; }
    public int StockQuantity { get; set; }
    public bool IsInStock { get; set; }
    public bool IsLowStock { get; set; }
    public bool IsActive { get; set; }
    public bool IsFeatured { get; set; }
    public string CategoryName { get; set; } = "";
    public string? BrandName { get; set; }
    public double AverageRating { get; set; }
    public int ReviewCount { get; set; }
    public IEnumerable<ProductImageDto> Images { get; set; } = Array.Empty<ProductImageDto>();
}

public class ProductImageDto
{
    public int Id { get; set; }
    public string Url { get; set; } = "";
    public string? AltText { get; set; }
    public bool IsPrimary { get; set; }
}

public class CreateProductRequest
{
    [Required]
    [StringLength(200)]
    public string Name { get; set; } = "";

    public string? Description { get; set; }
    public string? Sku { get; set; }

    [Range(0, double.MaxValue)]
    public decimal Price { get; set; }

    public decimal? CompareAtPrice { get; set; }

    [Range(0, int.MaxValue)]
    public int StockQuantity { get; set; }

    public int CategoryId { get; set; }
    public int? BrandId { get; set; }
}

public class UpdateProductRequest
{
    [Required]
    [StringLength(200)]
    public string Name { get; set; } = "";

    public string? Description { get; set; }
    public string? Sku { get; set; }
    public decimal Price { get; set; }
    public decimal? CompareAtPrice { get; set; }
    public int StockQuantity { get; set; }
    public int CategoryId { get; set; }
    public int? BrandId { get; set; }
    public bool IsActive { get; set; }
    public bool IsFeatured { get; set; }
}

public class UpdateInventoryRequest
{
    public int Quantity { get; set; }
}

public class ReviewDto
{
    public int Id { get; set; }
    public int Rating { get; set; }
    public string? Title { get; set; }
    public string? Content { get; set; }
    public string UserName { get; set; } = "";
    public bool IsVerifiedPurchase { get; set; }
    public DateTime CreatedAt { get; set; }
    public int HelpfulCount { get; set; }
}

public class CreateReviewRequest
{
    [Range(1, 5)]
    public int Rating { get; set; }

    public string? Title { get; set; }
    public string? Content { get; set; }
}

public class ProductQueryParams
{
    public int Page { get; set; } = 1;
    public int PageSize { get; set; } = 20;
    public string? SortBy { get; set; }
    public string? SortOrder { get; set; }
    public int? CategoryId { get; set; }
    public int? BrandId { get; set; }
    public decimal? MinPrice { get; set; }
    public decimal? MaxPrice { get; set; }
    public bool? InStock { get; set; }
}

public class OrderQueryParams
{
    public int Page { get; set; } = 1;
    public int PageSize { get; set; } = 20;
    public OrderStatus? Status { get; set; }
    public DateTime? FromDate { get; set; }
    public DateTime? ToDate { get; set; }
    public string? CustomerEmail { get; set; }
}

public class PagedResult<T>
{
    public IEnumerable<T> Items { get; set; } = Array.Empty<T>();
    public int Page { get; set; }
    public int PageSize { get; set; }
    public int TotalCount { get; set; }
    public int TotalPages { get; set; }
    public bool HasPrevious => Page > 1;
    public bool HasNext => Page < TotalPages;
}

public class PaginationParams
{
    public int Page { get; set; } = 1;
    public int PageSize { get; set; } = 10;
}

public class ErrorResponse
{
    public string Message { get; set; } = "";
    public string? Code { get; set; }
    public IDictionary<string, string[]>? Errors { get; set; }
}

public class PaymentRequest
{
    public Guid OrderId { get; set; }
    public decimal Amount { get; set; }
    public PaymentMethod PaymentMethod { get; set; }
    public string? CardToken { get; set; }
}

public class PaymentResultDto
{
    public bool Success { get; set; }
    public string? TransactionId { get; set; }
    public string? Message { get; set; }
    public string? ErrorCode { get; set; }
}

public class RefundResultDto
{
    public bool Success { get; set; }
    public string? RefundId { get; set; }
    public string? Message { get; set; }
}
