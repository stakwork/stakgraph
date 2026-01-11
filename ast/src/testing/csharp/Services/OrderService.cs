using CSharpTestServer.DTOs;
using CSharpTestServer.Models;
using CSharpTestServer.Repositories;
using AutoMapper;

namespace CSharpTestServer.Services;

public interface IOrderService
{
    Task<PagedResult<OrderDto>> GetAllAsync(OrderQueryParams queryParams);
    Task<OrderDto?> GetByIdAsync(Guid id);
    Task<IEnumerable<OrderDto>> GetByUserIdAsync(int userId);
    Task<OrderDto> CreateAsync(CreateOrderRequest request, int userId);
    Task ConfirmAsync(Guid id);
    Task CancelAsync(Guid id, string reason);
    Task MarkRefundedAsync(Guid id);
    Task<OrderDto> UpdateStatusAsync(Guid id, OrderStatus status);
    Task<IEnumerable<OrderItemDto>> GetOrderItemsAsync(Guid id);
    Task<OrderItemDto> AddItemAsync(Guid id, AddOrderItemRequest request);
    Task RemoveItemAsync(Guid id, int itemId);
    Task<TrackingInfoDto> GetTrackingInfoAsync(Guid id);
}

public class OrderService : IOrderService
{
    private readonly IOrderRepository _orderRepository;
    private readonly IProductRepository _productRepository;
    private readonly IPersonRepository _personRepository;
    private readonly IMapper _mapper;
    private readonly ILogger<OrderService> _logger;

    public OrderService(
        IOrderRepository orderRepository,
        IProductRepository productRepository,
        IPersonRepository personRepository,
        IMapper mapper,
        ILogger<OrderService> logger)
    {
        _orderRepository = orderRepository;
        _productRepository = productRepository;
        _personRepository = personRepository;
        _mapper = mapper;
        _logger = logger;
    }

    public async Task<PagedResult<OrderDto>> GetAllAsync(OrderQueryParams queryParams)
    {
        var (orders, totalCount) = await _orderRepository.GetAllAsync(queryParams);
        var items = _mapper.Map<IEnumerable<OrderDto>>(orders);
        
        return new PagedResult<OrderDto>
        {
            Items = items,
            Page = queryParams.Page,
            PageSize = queryParams.PageSize,
            TotalCount = totalCount,
            TotalPages = (int)Math.Ceiling(totalCount / (double)queryParams.PageSize)
        };
    }

    public async Task<OrderDto?> GetByIdAsync(Guid id)
    {
        var order = await _orderRepository.GetByIdWithItemsAsync(id);
        return order != null ? _mapper.Map<OrderDto>(order) : null;
    }

    public async Task<IEnumerable<OrderDto>> GetByUserIdAsync(int userId)
    {
        var orders = await _orderRepository.GetByUserIdAsync(userId);
        return _mapper.Map<IEnumerable<OrderDto>>(orders);
    }

    public async Task<OrderDto> CreateAsync(CreateOrderRequest request, int userId)
    {
        var customer = await _personRepository.GetByIdAsync(userId);
        if (customer == null)
        {
            throw new InvalidOperationException("Customer not found");
        }

        var order = new Order
        {
            OrderNumber = GenerateOrderNumber(),
            CustomerId = userId,
            CustomerEmail = customer.Email,
            ShippingAddress = _mapper.Map<Address>(request.ShippingAddress),
            BillingAddress = request.BillingAddress != null 
                ? _mapper.Map<Address>(request.BillingAddress) 
                : null,
            PaymentMethod = request.PaymentMethod,
            CouponCode = request.CouponCode
        };

        foreach (var itemRequest in request.Items)
        {
            var product = await _productRepository.GetByIdAsync(itemRequest.ProductId);
            if (product == null)
            {
                throw new InvalidOperationException($"Product {itemRequest.ProductId} not found");
            }

            if (product.StockQuantity < itemRequest.Quantity)
            {
                throw new InvalidOperationException($"Insufficient stock for product {product.Name}");
            }

            order.Items.Add(new OrderItem
            {
                ProductId = product.Id,
                ProductName = product.Name,
                ProductSku = product.Sku,
                Quantity = itemRequest.Quantity,
                UnitPrice = product.Price
            });

            product.DecrementStock(itemRequest.Quantity);
            await _productRepository.UpdateAsync(product);
        }

        order.CalculateTotals();
        await _orderRepository.AddAsync(order);

        _logger.LogInformation("Order {OrderNumber} created for user {UserId}", order.OrderNumber, userId);
        return _mapper.Map<OrderDto>(order);
    }

    public async Task ConfirmAsync(Guid id)
    {
        var order = await _orderRepository.GetByIdAsync(id);
        if (order == null)
        {
            throw new KeyNotFoundException($"Order {id} not found");
        }

        order.Confirm();
        await _orderRepository.UpdateAsync(order);
        
        _logger.LogInformation("Order {Id} confirmed", id);
    }

    public async Task CancelAsync(Guid id, string reason)
    {
        var order = await _orderRepository.GetByIdWithItemsAsync(id);
        if (order == null)
        {
            throw new KeyNotFoundException($"Order {id} not found");
        }

        foreach (var item in order.Items)
        {
            var product = await _productRepository.GetByIdAsync(item.ProductId);
            if (product != null)
            {
                product.IncrementStock(item.Quantity);
                await _productRepository.UpdateAsync(product);
            }
        }

        order.Cancel(reason);
        await _orderRepository.UpdateAsync(order);
        
        _logger.LogInformation("Order {Id} cancelled: {Reason}", id, reason);
    }

    public async Task MarkRefundedAsync(Guid id)
    {
        var order = await _orderRepository.GetByIdAsync(id);
        if (order == null)
        {
            throw new KeyNotFoundException($"Order {id} not found");
        }

        order.Status = OrderStatus.Refunded;
        await _orderRepository.UpdateAsync(order);
    }

    public async Task<OrderDto> UpdateStatusAsync(Guid id, OrderStatus status)
    {
        var order = await _orderRepository.GetByIdAsync(id);
        if (order == null)
        {
            throw new KeyNotFoundException($"Order {id} not found");
        }

        order.Status = status;
        await _orderRepository.UpdateAsync(order);
        
        return _mapper.Map<OrderDto>(order);
    }

    public async Task<IEnumerable<OrderItemDto>> GetOrderItemsAsync(Guid id)
    {
        var order = await _orderRepository.GetByIdWithItemsAsync(id);
        if (order == null)
        {
            throw new KeyNotFoundException($"Order {id} not found");
        }

        return _mapper.Map<IEnumerable<OrderItemDto>>(order.Items);
    }

    public async Task<OrderItemDto> AddItemAsync(Guid id, AddOrderItemRequest request)
    {
        var order = await _orderRepository.GetByIdWithItemsAsync(id);
        if (order == null)
        {
            throw new KeyNotFoundException($"Order {id} not found");
        }

        var product = await _productRepository.GetByIdAsync(request.ProductId);
        if (product == null)
        {
            throw new InvalidOperationException($"Product {request.ProductId} not found");
        }

        var item = new OrderItem
        {
            OrderId = id,
            ProductId = product.Id,
            ProductName = product.Name,
            ProductSku = product.Sku,
            Quantity = request.Quantity,
            UnitPrice = product.Price
        };

        order.Items.Add(item);
        order.CalculateTotals();
        await _orderRepository.UpdateAsync(order);

        return _mapper.Map<OrderItemDto>(item);
    }

    public async Task RemoveItemAsync(Guid id, int itemId)
    {
        var order = await _orderRepository.GetByIdWithItemsAsync(id);
        if (order == null)
        {
            throw new KeyNotFoundException($"Order {id} not found");
        }

        var item = order.Items.FirstOrDefault(i => i.Id == itemId);
        if (item != null)
        {
            order.Items.Remove(item);
            order.CalculateTotals();
            await _orderRepository.UpdateAsync(order);
        }
    }

    public async Task<TrackingInfoDto> GetTrackingInfoAsync(Guid id)
    {
        var order = await _orderRepository.GetByIdAsync(id);
        if (order == null)
        {
            throw new KeyNotFoundException($"Order {id} not found");
        }

        return _mapper.Map<TrackingInfoDto>(order.Tracking);
    }

    private string GenerateOrderNumber()
    {
        return $"ORD-{DateTime.UtcNow:yyyyMMdd}-{Guid.NewGuid().ToString("N")[..8].ToUpper()}";
    }
}
