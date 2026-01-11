using Xunit;
using Moq;
using CSharpTestServer.Services;
using CSharpTestServer.Repositories;
using CSharpTestServer.DTOs;
using CSharpTestServer.Models;
using AutoMapper;
using Microsoft.Extensions.Logging;

namespace CSharpTestServer.Tests.Unit;

public class PersonServiceTests
{
    private readonly Mock<IPersonRepository> _personRepositoryMock;
    private readonly Mock<IArticleRepository> _articleRepositoryMock;
    private readonly Mock<IFileStorageService> _fileStorageMock;
    private readonly Mock<IMapper> _mapperMock;
    private readonly Mock<ILogger<PersonService>> _loggerMock;
    private readonly PersonService _service;

    public PersonServiceTests()
    {
        _personRepositoryMock = new Mock<IPersonRepository>();
        _articleRepositoryMock = new Mock<IArticleRepository>();
        _fileStorageMock = new Mock<IFileStorageService>();
        _mapperMock = new Mock<IMapper>();
        _loggerMock = new Mock<ILogger<PersonService>>();

        _service = new PersonService(
            _personRepositoryMock.Object,
            _articleRepositoryMock.Object,
            _fileStorageMock.Object,
            _mapperMock.Object,
            _loggerMock.Object);
    }

    [Fact]
    public async Task GetAllAsync_ReturnsPagedResults()
    {
        var persons = new List<Person>
        {
            new() { Id = 1, FirstName = "John", LastName = "Doe", Email = "john@example.com" },
            new() { Id = 2, FirstName = "Jane", LastName = "Smith", Email = "jane@example.com" }
        };
        var personDtos = persons.Select(p => new PersonDto { Id = p.Id, FirstName = p.FirstName }).ToList();

        _personRepositoryMock.Setup(r => r.GetAllAsync(1, 10)).ReturnsAsync(persons);
        _mapperMock.Setup(m => m.Map<IEnumerable<PersonDto>>(persons)).Returns(personDtos);

        var result = await _service.GetAllAsync(1, 10);

        Assert.NotNull(result);
        Assert.Equal(2, result.Count());
    }

    [Fact]
    public async Task GetByIdAsync_WhenPersonExists_ReturnsPerson()
    {
        var person = new Person { Id = 1, FirstName = "John", LastName = "Doe", Email = "john@example.com" };
        var personDto = new PersonDto { Id = 1, FirstName = "John" };

        _personRepositoryMock.Setup(r => r.GetByIdAsync(1)).ReturnsAsync(person);
        _mapperMock.Setup(m => m.Map<PersonDto>(person)).Returns(personDto);

        var result = await _service.GetByIdAsync(1);

        Assert.NotNull(result);
        Assert.Equal(1, result.Id);
    }

    [Fact]
    public async Task GetByIdAsync_WhenPersonDoesNotExist_ReturnsNull()
    {
        _personRepositoryMock.Setup(r => r.GetByIdAsync(999)).ReturnsAsync((Person?)null);

        var result = await _service.GetByIdAsync(999);

        Assert.Null(result);
    }

    [Fact]
    public async Task CreateAsync_CreatesNewPerson()
    {
        var request = new CreatePersonRequest
        {
            FirstName = "John",
            LastName = "Doe",
            Email = "john@example.com",
            DateOfBirth = new DateTime(1990, 1, 1)
        };

        _personRepositoryMock.Setup(r => r.AddAsync(It.IsAny<Person>())).Returns(Task.CompletedTask);
        _mapperMock.Setup(m => m.Map<PersonDto>(It.IsAny<Person>()))
            .Returns(new PersonDto { Id = 1, FirstName = "John" });

        var result = await _service.CreateAsync(request);

        Assert.NotNull(result);
        _personRepositoryMock.Verify(r => r.AddAsync(It.IsAny<Person>()), Times.Once);
    }

    [Fact]
    public async Task UpdateAsync_WhenPersonExists_UpdatesPerson()
    {
        var person = new Person { Id = 1, FirstName = "John", LastName = "Doe", Email = "john@example.com" };
        var request = new UpdatePersonRequest
        {
            FirstName = "John Updated",
            LastName = "Doe",
            Email = "john@example.com",
            DateOfBirth = new DateTime(1990, 1, 1)
        };

        _personRepositoryMock.Setup(r => r.GetByIdAsync(1)).ReturnsAsync(person);
        _personRepositoryMock.Setup(r => r.UpdateAsync(It.IsAny<Person>())).Returns(Task.CompletedTask);
        _mapperMock.Setup(m => m.Map<PersonDto>(It.IsAny<Person>()))
            .Returns(new PersonDto { Id = 1, FirstName = "John Updated" });

        var result = await _service.UpdateAsync(1, request);

        Assert.NotNull(result);
        Assert.Equal("John Updated", result.FirstName);
    }

    [Fact]
    public async Task DeleteAsync_WhenPersonExists_ReturnsTrue()
    {
        var person = new Person { Id = 1, FirstName = "John", LastName = "Doe", Email = "john@example.com" };

        _personRepositoryMock.Setup(r => r.GetByIdAsync(1)).ReturnsAsync(person);
        _personRepositoryMock.Setup(r => r.DeleteAsync(1)).Returns(Task.CompletedTask);

        var result = await _service.DeleteAsync(1);

        Assert.True(result);
        _personRepositoryMock.Verify(r => r.DeleteAsync(1), Times.Once);
    }

    [Fact]
    public async Task DeleteAsync_WhenPersonDoesNotExist_ReturnsFalse()
    {
        _personRepositoryMock.Setup(r => r.GetByIdAsync(999)).ReturnsAsync((Person?)null);

        var result = await _service.DeleteAsync(999);

        Assert.False(result);
        _personRepositoryMock.Verify(r => r.DeleteAsync(It.IsAny<int>()), Times.Never);
    }

    [Fact]
    public async Task SearchAsync_ReturnsMatchingPersons()
    {
        var persons = new List<Person>
        {
            new() { Id = 1, FirstName = "John", LastName = "Doe", Email = "john@example.com" }
        };

        _personRepositoryMock.Setup(r => r.SearchAsync("john")).ReturnsAsync(persons);
        _mapperMock.Setup(m => m.Map<IEnumerable<PersonDto>>(persons))
            .Returns(new List<PersonDto> { new() { Id = 1, FirstName = "John" } });

        var result = await _service.SearchAsync("john");

        Assert.Single(result);
    }
}

public class ArticleServiceTests
{
    private readonly Mock<IArticleRepository> _articleRepositoryMock;
    private readonly Mock<IMapper> _mapperMock;
    private readonly Mock<ILogger<ArticleService>> _loggerMock;
    private readonly ArticleService _service;

    public ArticleServiceTests()
    {
        _articleRepositoryMock = new Mock<IArticleRepository>();
        _mapperMock = new Mock<IMapper>();
        _loggerMock = new Mock<ILogger<ArticleService>>();

        _service = new ArticleService(
            _articleRepositoryMock.Object,
            _mapperMock.Object,
            _loggerMock.Object);
    }

    [Fact]
    public async Task GetByIdAsync_IncrementsViewCount()
    {
        var article = new Article { Id = 1, Title = "Test Article", Content = "Content", ViewCount = 0 };
        var articleDto = new ArticleDto { Id = 1, Title = "Test Article", ViewCount = 1 };

        _articleRepositoryMock.Setup(r => r.GetByIdWithDetailsAsync(1)).ReturnsAsync(article);
        _articleRepositoryMock.Setup(r => r.UpdateAsync(article)).Returns(Task.CompletedTask);
        _mapperMock.Setup(m => m.Map<ArticleDto>(article)).Returns(articleDto);

        await _service.GetByIdAsync(1);

        Assert.Equal(1, article.ViewCount);
        _articleRepositoryMock.Verify(r => r.UpdateAsync(article), Times.Once);
    }

    [Fact]
    public async Task CreateAsync_GeneratesSlug()
    {
        var request = new CreateArticleRequest
        {
            Title = "Test Article Title",
            Content = "Content"
        };

        _articleRepositoryMock.Setup(r => r.AddAsync(It.IsAny<Article>())).Returns(Task.CompletedTask);
        _mapperMock.Setup(m => m.Map<ArticleDto>(It.IsAny<Article>()))
            .Returns(new ArticleDto { Id = 1, Title = "Test Article Title" });

        await _service.CreateAsync(request, 1);

        _articleRepositoryMock.Verify(r => r.AddAsync(It.Is<Article>(a => 
            a.Slug == "test-article-title")), Times.Once);
    }

    [Fact]
    public async Task PublishAsync_SetsPublishedStatus()
    {
        var article = new Article { Id = 1, Title = "Test", Content = "Content", Status = ArticleStatus.Draft };

        _articleRepositoryMock.Setup(r => r.GetByIdAsync(1)).ReturnsAsync(article);
        _articleRepositoryMock.Setup(r => r.UpdateAsync(article)).Returns(Task.CompletedTask);
        _mapperMock.Setup(m => m.Map<ArticleDto>(article)).Returns(new ArticleDto { Id = 1 });

        await _service.PublishAsync(1);

        Assert.Equal(ArticleStatus.Published, article.Status);
        Assert.NotNull(article.PublishedAt);
    }

    [Fact]
    public async Task GetFeaturedAsync_ReturnsFeaturedArticles()
    {
        var articles = new List<Article>
        {
            new() { Id = 1, Title = "Featured 1", Content = "Content", IsFeatured = true },
            new() { Id = 2, Title = "Featured 2", Content = "Content", IsFeatured = true }
        };

        _articleRepositoryMock.Setup(r => r.GetFeaturedAsync()).ReturnsAsync(articles);
        _mapperMock.Setup(m => m.Map<IEnumerable<ArticleDto>>(articles))
            .Returns(articles.Select(a => new ArticleDto { Id = a.Id }).ToList());

        var result = await _service.GetFeaturedAsync();

        Assert.Equal(2, result.Count());
    }
}

public class OrderServiceTests
{
    private readonly Mock<IOrderRepository> _orderRepositoryMock;
    private readonly Mock<IProductRepository> _productRepositoryMock;
    private readonly Mock<IPersonRepository> _personRepositoryMock;
    private readonly Mock<IMapper> _mapperMock;
    private readonly Mock<ILogger<OrderService>> _loggerMock;
    private readonly OrderService _service;

    public OrderServiceTests()
    {
        _orderRepositoryMock = new Mock<IOrderRepository>();
        _productRepositoryMock = new Mock<IProductRepository>();
        _personRepositoryMock = new Mock<IPersonRepository>();
        _mapperMock = new Mock<IMapper>();
        _loggerMock = new Mock<ILogger<OrderService>>();

        _service = new OrderService(
            _orderRepositoryMock.Object,
            _productRepositoryMock.Object,
            _personRepositoryMock.Object,
            _mapperMock.Object,
            _loggerMock.Object);
    }

    [Fact]
    public async Task CreateAsync_CalculatesTotals()
    {
        var customer = new Person { Id = 1, Email = "test@example.com" };
        var product = new Product { Id = 1, Name = "Product", Price = 100, StockQuantity = 10 };
        var request = new CreateOrderRequest
        {
            Items = new List<CreateOrderItemRequest>
            {
                new() { ProductId = 1, Quantity = 2 }
            },
            ShippingAddress = new AddressDto { Street = "123 Main St", City = "NYC", State = "NY", PostalCode = "10001", Country = "US" },
            PaymentMethod = PaymentMethod.CreditCard
        };

        _personRepositoryMock.Setup(r => r.GetByIdAsync(1)).ReturnsAsync(customer);
        _productRepositoryMock.Setup(r => r.GetByIdAsync(1)).ReturnsAsync(product);
        _productRepositoryMock.Setup(r => r.UpdateAsync(product)).Returns(Task.CompletedTask);
        _orderRepositoryMock.Setup(r => r.AddAsync(It.IsAny<Order>())).Returns(Task.CompletedTask);
        _mapperMock.Setup(m => m.Map<Address>(It.IsAny<AddressDto>())).Returns(new Address());
        _mapperMock.Setup(m => m.Map<OrderDto>(It.IsAny<Order>())).Returns(new OrderDto { Id = Guid.NewGuid() });

        await _service.CreateAsync(request, 1);

        _orderRepositoryMock.Verify(r => r.AddAsync(It.Is<Order>(o => 
            o.SubTotal == 200 && o.Items.Count == 1)), Times.Once);
    }

    [Fact]
    public async Task ConfirmAsync_SetsConfirmedStatus()
    {
        var orderId = Guid.NewGuid();
        var order = new Order { Id = orderId, Status = OrderStatus.Pending };

        _orderRepositoryMock.Setup(r => r.GetByIdAsync(orderId)).ReturnsAsync(order);
        _orderRepositoryMock.Setup(r => r.UpdateAsync(order)).Returns(Task.CompletedTask);

        await _service.ConfirmAsync(orderId);

        Assert.Equal(OrderStatus.Confirmed, order.Status);
        Assert.NotNull(order.PaidAt);
    }

    [Fact]
    public async Task CancelAsync_RestoresInventory()
    {
        var orderId = Guid.NewGuid();
        var product = new Product { Id = 1, StockQuantity = 5 };
        var order = new Order
        {
            Id = orderId,
            Status = OrderStatus.Confirmed,
            Items = new List<OrderItem>
            {
                new() { ProductId = 1, Quantity = 2 }
            }
        };

        _orderRepositoryMock.Setup(r => r.GetByIdWithItemsAsync(orderId)).ReturnsAsync(order);
        _productRepositoryMock.Setup(r => r.GetByIdAsync(1)).ReturnsAsync(product);
        _productRepositoryMock.Setup(r => r.UpdateAsync(product)).Returns(Task.CompletedTask);
        _orderRepositoryMock.Setup(r => r.UpdateAsync(order)).Returns(Task.CompletedTask);

        await _service.CancelAsync(orderId, "Customer request");

        Assert.Equal(7, product.StockQuantity);
        Assert.Equal(OrderStatus.Cancelled, order.Status);
    }
}

public class ProductModelTests
{
    [Fact]
    public void IsInStock_WhenQuantityGreaterThanZero_ReturnsTrue()
    {
        var product = new Product { StockQuantity = 10 };

        Assert.True(product.IsInStock);
    }

    [Fact]
    public void IsInStock_WhenQuantityIsZero_ReturnsFalse()
    {
        var product = new Product { StockQuantity = 0 };

        Assert.False(product.IsInStock);
    }

    [Fact]
    public void IsLowStock_WhenBelowThreshold_ReturnsTrue()
    {
        var product = new Product { StockQuantity = 5, LowStockThreshold = 10 };

        Assert.True(product.IsLowStock);
    }

    [Fact]
    public void DiscountPercentage_CalculatesCorrectly()
    {
        var product = new Product { Price = 80, CompareAtPrice = 100 };

        Assert.Equal(20, product.DiscountPercentage);
    }

    [Fact]
    public void DecrementStock_ThrowsWhenInsufficientStock()
    {
        var product = new Product { StockQuantity = 5 };

        Assert.Throws<InvalidOperationException>(() => product.DecrementStock(10));
    }

    [Fact]
    public void DecrementStock_ReducesQuantity()
    {
        var product = new Product { StockQuantity = 10 };

        product.DecrementStock(3);

        Assert.Equal(7, product.StockQuantity);
    }
}

public class PersonModelTests
{
    [Fact]
    public void FullName_ConcatenatesFirstAndLastName()
    {
        var person = new Person { FirstName = "John", LastName = "Doe" };

        Assert.Equal("John Doe", person.FullName);
    }

    [Fact]
    public void CalculateAge_ReturnsCorrectAge()
    {
        var today = DateTime.Today;
        var person = new Person { DateOfBirth = today.AddYears(-30) };

        Assert.Equal(30, person.CalculateAge());
    }

    [Fact]
    public void Deactivate_SetsIsActiveFalse()
    {
        var person = new Person { IsActive = true };

        person.Deactivate();

        Assert.False(person.IsActive);
        Assert.NotNull(person.UpdatedAt);
    }
}

public class ArticleModelTests
{
    [Fact]
    public void Publish_SetsCorrectState()
    {
        var article = new Article { Status = ArticleStatus.Draft };

        article.Publish();

        Assert.Equal(ArticleStatus.Published, article.Status);
        Assert.NotNull(article.PublishedAt);
    }

    [Fact]
    public void Unpublish_SetsCorrectState()
    {
        var article = new Article { Status = ArticleStatus.Published, PublishedAt = DateTime.UtcNow };

        article.Unpublish();

        Assert.Equal(ArticleStatus.Draft, article.Status);
        Assert.Null(article.PublishedAt);
    }

    [Fact]
    public void GenerateSlug_CreatesValidSlug()
    {
        var article = new Article { Title = "Hello World, This is a Test!" };

        var slug = article.GenerateSlug();

        Assert.Equal("hello-world-this-is-a-test!", slug);
    }

    [Fact]
    public void IncrementViewCount_IncreasesCount()
    {
        var article = new Article { ViewCount = 0 };

        article.IncrementViewCount();

        Assert.Equal(1, article.ViewCount);
    }
}
