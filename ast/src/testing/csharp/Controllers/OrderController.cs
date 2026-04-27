// @ast node: Class "OrderController"
// @ast node: Function "AddItem"
// @ast node: Function "Cancel"
// @ast node: Function "Checkout"
// @ast node: Function "Create"
// @ast node: Function "GetAll"
// @ast node: Function "GetById"
// @ast node: Function "GetCurrentUserId"
// @ast node: Function "GetMyOrders"
// @ast node: Function "GetOrderItems"
// @ast node: Function "GetTracking"
// @ast node: Function "OrderController"
// @ast node: Function "Refund"
// @ast node: Function "RemoveItem"
// @ast node: Function "UpdateStatus"
// @ast node: Endpoint "Create"
// @ast edge: Handler -> Function "Create" "OrderController.cs"
// @ast node: Endpoint "GetAll"
// @ast edge: Handler -> Function "GetAll" "OrderController.cs"
// @ast node: Endpoint "my-orders"
// @ast edge: Handler -> Function "GetMyOrders" "OrderController.cs"
// @ast node: Endpoint "{id:guid}"
// @ast edge: Handler -> Function "GetById" "OrderController.cs"
// @ast node: Endpoint "{id:guid}/cancel"
// @ast edge: Handler -> Function "Cancel" "OrderController.cs"
// @ast node: Endpoint "{id:guid}/checkout"
// @ast edge: Handler -> Function "Checkout" "OrderController.cs"
// @ast node: Endpoint "{id:guid}/items"
// @ast edge: Handler -> Function "GetOrderItems" "OrderController.cs"
// @ast node: Endpoint "{id:guid}/items"
// @ast node: Endpoint "{id:guid}/items/{itemId:int}"
// @ast edge: Handler -> Function "RemoveItem" "OrderController.cs"
// @ast node: Endpoint "{id:guid}/refund"
// @ast edge: Handler -> Function "Refund" "OrderController.cs"
// @ast node: Endpoint "{id:guid}/status"
// @ast edge: Handler -> Function "UpdateStatus" "OrderController.cs"
// @ast node: Endpoint "{id:guid}/tracking"
// @ast edge: Handler -> Function "GetTracking" "OrderController.cs"
// @ast node: Var "_logger"
// @ast node: Var "_notificationService"
// @ast node: Var "_orderService"
// @ast node: Var "_paymentService"
// @ast node: Import "import-imports-srctestingcsharpcontrollersordercontrollercs-43"
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Authorization;
using CSharpTestServer.DTOs;
using CSharpTestServer.Services;

namespace CSharpTestServer.Controllers;

[ApiController]
[Route("api/[controller]")]
public class OrderController : ControllerBase
{
    private readonly IOrderService _orderService;
    private readonly IPaymentService _paymentService;
    private readonly INotificationService _notificationService;
    private readonly ILogger<OrderController> _logger;

    public OrderController(
        IOrderService orderService,
        IPaymentService paymentService,
        INotificationService notificationService,
        ILogger<OrderController> logger)
    {
        _orderService = orderService;
        _paymentService = paymentService;
        _notificationService = notificationService;
        _logger = logger;
    }

    [HttpGet]
    [Authorize(Policy = "AdminOnly")]
    public async Task<ActionResult<PagedResult<OrderDto>>> GetAll([FromQuery] OrderQueryParams queryParams)
    {
        var orders = await _orderService.GetAllAsync(queryParams);
        return Ok(orders);
    }

    [HttpGet("{id:guid}")]
    [Authorize]
    public async Task<ActionResult<OrderDto>> GetById(Guid id)
    {
        var order = await _orderService.GetByIdAsync(id);
        if (order == null)
        {
            return NotFound();
        }
        return Ok(order);
    }

    [HttpGet("my-orders")]
    [Authorize]
    public async Task<ActionResult<IEnumerable<OrderDto>>> GetMyOrders()
    {
        var userId = GetCurrentUserId();
        var orders = await _orderService.GetByUserIdAsync(userId);
        return Ok(orders);
    }

    [HttpPost]
    [Authorize]
    public async Task<ActionResult<OrderDto>> Create([FromBody] CreateOrderRequest request)
    {
        var userId = GetCurrentUserId();
        var order = await _orderService.CreateAsync(request, userId);
        
        _logger.LogInformation("Order {OrderId} created for user {UserId}", order.Id, userId);
        
        return CreatedAtAction(nameof(GetById), new { id = order.Id }, order);
    }

    [HttpPost("{id:guid}/checkout")]
    [Authorize]
    public async Task<ActionResult<PaymentResultDto>> Checkout(Guid id, [FromBody] CheckoutRequest request)
    {
        var order = await _orderService.GetByIdAsync(id);
        if (order == null)
        {
            return NotFound();
        }

        var paymentResult = await _paymentService.ProcessPaymentAsync(new PaymentRequest
        {
            OrderId = id,
            Amount = order.TotalAmount,
            PaymentMethod = request.PaymentMethod,
            CardToken = request.CardToken
        });

        if (paymentResult.Success)
        {
            await _orderService.ConfirmAsync(id);
            await _notificationService.SendOrderConfirmationAsync(order.CustomerEmail, order.Id);
        }

        return Ok(paymentResult);
    }

    [HttpPost("{id:guid}/cancel")]
    [Authorize]
    public async Task<ActionResult> Cancel(Guid id, [FromBody] CancelOrderRequest request)
    {
        var order = await _orderService.GetByIdAsync(id);
        if (order == null)
        {
            return NotFound();
        }

        await _orderService.CancelAsync(id, request.Reason);
        await _notificationService.SendOrderCancelledAsync(order.CustomerEmail, id);
        
        return Ok();
    }

    [HttpPost("{id:guid}/refund")]
    [Authorize(Policy = "AdminOnly")]
    public async Task<ActionResult<RefundResultDto>> Refund(Guid id)
    {
        var result = await _paymentService.RefundAsync(id);
        if (result.Success)
        {
            await _orderService.MarkRefundedAsync(id);
        }
        return Ok(result);
    }

    [HttpPatch("{id:guid}/status")]
    [Authorize(Policy = "AdminOnly")]
    public async Task<ActionResult<OrderDto>> UpdateStatus(Guid id, [FromBody] UpdateOrderStatusRequest request)
    {
        var order = await _orderService.UpdateStatusAsync(id, request.Status);
        return Ok(order);
    }

    [HttpGet("{id:guid}/items")]
    public async Task<ActionResult<IEnumerable<OrderItemDto>>> GetOrderItems(Guid id)
    {
        var items = await _orderService.GetOrderItemsAsync(id);
        return Ok(items);
    }

    [HttpPost("{id:guid}/items")]
    [Authorize]
    public async Task<ActionResult<OrderItemDto>> AddItem(Guid id, [FromBody] AddOrderItemRequest request)
    {
        var item = await _orderService.AddItemAsync(id, request);
        return Ok(item);
    }

    [HttpDelete("{id:guid}/items/{itemId:int}")]
    [Authorize]
    public async Task<ActionResult> RemoveItem(Guid id, int itemId)
    {
        await _orderService.RemoveItemAsync(id, itemId);
        return NoContent();
    }

    [HttpGet("{id:guid}/tracking")]
    public async Task<ActionResult<TrackingInfoDto>> GetTracking(Guid id)
    {
        var tracking = await _orderService.GetTrackingInfoAsync(id);
        return Ok(tracking);
    }

    private int GetCurrentUserId()
    {
        var claim = User.FindFirst("sub");
        return int.Parse(claim?.Value ?? "0");
    }
}
