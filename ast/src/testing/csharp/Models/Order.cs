using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace CSharpTestServer.Models;

[Table("Orders")]
public class Order
{
    public Guid Id { get; set; } = Guid.NewGuid();

    [Required]
    public string OrderNumber { get; set; } = "";

    public int CustomerId { get; set; }

    [ForeignKey("CustomerId")]
    public Person Customer { get; set; } = null!;

    public string CustomerEmail { get; set; } = "";

    public OrderStatus Status { get; set; } = OrderStatus.Pending;

    public decimal SubTotal { get; set; }

    public decimal TaxAmount { get; set; }

    public decimal ShippingCost { get; set; }

    public decimal DiscountAmount { get; set; }

    public decimal TotalAmount { get; set; }

    public string? CouponCode { get; set; }

    public PaymentMethod PaymentMethod { get; set; }

    public string? PaymentTransactionId { get; set; }

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    public DateTime? PaidAt { get; set; }

    public DateTime? ShippedAt { get; set; }

    public DateTime? DeliveredAt { get; set; }

    public DateTime? CancelledAt { get; set; }

    public string? CancellationReason { get; set; }

    public Address ShippingAddress { get; set; } = new();

    public Address? BillingAddress { get; set; }

    public ICollection<OrderItem> Items { get; set; } = new List<OrderItem>();

    public TrackingInfo? Tracking { get; set; }

    public void Confirm()
    {
        Status = OrderStatus.Confirmed;
        PaidAt = DateTime.UtcNow;
    }

    public void Ship(string trackingNumber, string carrier)
    {
        Status = OrderStatus.Shipped;
        ShippedAt = DateTime.UtcNow;
        Tracking = new TrackingInfo
        {
            TrackingNumber = trackingNumber,
            Carrier = carrier,
            EstimatedDelivery = DateTime.UtcNow.AddDays(5)
        };
    }

    public void Deliver()
    {
        Status = OrderStatus.Delivered;
        DeliveredAt = DateTime.UtcNow;
    }

    public void Cancel(string reason)
    {
        Status = OrderStatus.Cancelled;
        CancelledAt = DateTime.UtcNow;
        CancellationReason = reason;
    }

    public void CalculateTotals()
    {
        SubTotal = Items.Sum(i => i.TotalPrice);
        TaxAmount = SubTotal * 0.1m;
        TotalAmount = SubTotal + TaxAmount + ShippingCost - DiscountAmount;
    }
}

public enum OrderStatus
{
    Pending,
    Confirmed,
    Processing,
    Shipped,
    Delivered,
    Cancelled,
    Refunded
}

public enum PaymentMethod
{
    CreditCard,
    DebitCard,
    PayPal,
    BankTransfer,
    CashOnDelivery
}

public class OrderItem
{
    public int Id { get; set; }

    public Guid OrderId { get; set; }

    [ForeignKey("OrderId")]
    public Order Order { get; set; } = null!;

    public int ProductId { get; set; }

    [ForeignKey("ProductId")]
    public Product Product { get; set; } = null!;

    public string ProductName { get; set; } = "";

    public string? ProductSku { get; set; }

    public int Quantity { get; set; }

    public decimal UnitPrice { get; set; }

    public decimal TotalPrice => Quantity * UnitPrice;
}

public class Address
{
    public string Street { get; set; } = "";
    public string City { get; set; } = "";
    public string State { get; set; } = "";
    public string PostalCode { get; set; } = "";
    public string Country { get; set; } = "";
}

public class TrackingInfo
{
    public string TrackingNumber { get; set; } = "";
    public string Carrier { get; set; } = "";
    public DateTime? EstimatedDelivery { get; set; }
    public List<TrackingEvent> Events { get; set; } = new();
}

public class TrackingEvent
{
    public DateTime Timestamp { get; set; }
    public string Status { get; set; } = "";
    public string Location { get; set; } = "";
    public string? Description { get; set; }
}
