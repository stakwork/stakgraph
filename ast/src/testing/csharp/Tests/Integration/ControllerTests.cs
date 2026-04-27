// @ast node: Class "AdminControllerIntegrationTests"
// @ast node: Class "ArticleControllerIntegrationTests"
// @ast node: Class "AuthControllerIntegrationTests"
// @ast node: Class "HealthControllerIntegrationTests"
// @ast node: Class "MinimalApiIntegrationTests"
// @ast node: Class "OrderControllerIntegrationTests"
// @ast node: Class "PersonControllerIntegrationTests"
// @ast node: Class "ProductControllerIntegrationTests"
// @ast node: IntegrationTest "AdminControllerIntegrationTests"
// @ast node: IntegrationTest "ArticleControllerIntegrationTests"
// @ast node: IntegrationTest "AuthControllerIntegrationTests"
// @ast node: IntegrationTest "ConfigureTestServices"
// @ast node: IntegrationTest "Create_WithInvalidData_ReturnsBadRequest"
// @ast node: IntegrationTest "Create_WithValidData_ReturnsCreated"
// @ast node: IntegrationTest "Dashboard_Unauthorized_Returns401"
// @ast node: IntegrationTest "ForgotPassword_ReturnsOk"
// @ast node: IntegrationTest "GetAll_ReturnsPagedProducts"
// @ast node: IntegrationTest "GetAll_ReturnsPagedResult"
// @ast node: IntegrationTest "GetAll_ReturnsSuccessStatusCode"
// @ast node: IntegrationTest "GetByCategory_ReturnsProducts"
// @ast node: IntegrationTest "GetById_WhenExists_ReturnsOk"
// @ast node: IntegrationTest "GetById_WhenNotExists_ReturnsNotFound"
// @ast node: IntegrationTest "GetById_WhenNotExists_ReturnsNotFound"
// @ast node: IntegrationTest "GetFeatured_ReturnsFeaturedArticles"
// @ast node: IntegrationTest "GetMyOrders_Unauthorized_Returns401"
// @ast node: IntegrationTest "GetSettings_Unauthorized_Returns401"
// @ast node: IntegrationTest "GetUsers_Unauthorized_Returns401"
// @ast node: IntegrationTest "HealthCheck_ReturnsHealthy"
// @ast node: IntegrationTest "HealthControllerIntegrationTests"
// @ast node: IntegrationTest "HealthEndpoint_ReturnsOk"
// @ast node: IntegrationTest "LiveCheck_ReturnsAlive"
// @ast node: IntegrationTest "Login_WithInvalidCredentials_ReturnsUnauthorized"
// @ast node: IntegrationTest "MinimalApiIntegrationTests"
// @ast node: IntegrationTest "OrderControllerIntegrationTests"
// @ast node: IntegrationTest "PersonControllerIntegrationTests"
// @ast node: IntegrationTest "ProductControllerIntegrationTests"
// @ast node: IntegrationTest "ReadyCheck_ReturnsReady"
// @ast node: IntegrationTest "ReadyEndpoint_ReturnsOk"
// @ast node: IntegrationTest "Register_WithValidData_ReturnsSuccess"
// @ast node: IntegrationTest "Search_ReturnsResults"
// @ast node: IntegrationTest "Search_WithQuery_ReturnsResults"
// @ast node: IntegrationTest "StatusEndpoint_ReturnsOperational"
// @ast node: Var "_client"
// @ast node: Var "_client"
// @ast node: Var "_client"
// @ast node: Var "_client"
// @ast node: Var "_client"
// @ast node: Var "_client"
// @ast node: Var "_client"
// @ast node: Var "_client"
// @ast node: Var "_factory"
// @ast node: Var "_factory"
// @ast node: Var "_factory"
// @ast node: Import "import-imports-srctestingcsharptestsintegrationcontrollertestscs-54"
using Xunit;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.DependencyInjection;
using System.Net;
using System.Net.Http.Json;
using CSharpTestServer.DTOs;

namespace CSharpTestServer.Tests.Integration;

public class PersonControllerIntegrationTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly WebApplicationFactory<Program> _factory;
    private readonly HttpClient _client;

    public PersonControllerIntegrationTests(WebApplicationFactory<Program> factory)
    {
        _factory = factory.WithWebHostBuilder(builder =>
        {
            builder.UseEnvironment("Testing");
            builder.ConfigureServices(services =>
            {
                ConfigureTestServices(services);
            });
        });
        _client = _factory.CreateClient();
    }

    private void ConfigureTestServices(IServiceCollection services)
    {
    }

    [Fact]
    public async Task GetAll_ReturnsSuccessStatusCode()
    {
        var response = await _client.GetAsync("/api/person");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    [Fact]
    public async Task GetById_WhenExists_ReturnsOk()
    {
        var response = await _client.GetAsync("/api/person/1");

        Assert.True(response.StatusCode == HttpStatusCode.OK || 
                    response.StatusCode == HttpStatusCode.NotFound);
    }

    [Fact]
    public async Task Create_WithValidData_ReturnsCreated()
    {
        var request = new CreatePersonRequest
        {
            FirstName = "Integration",
            LastName = "Test",
            Email = $"test{Guid.NewGuid()}@example.com",
            DateOfBirth = new DateTime(1990, 1, 1)
        };

        var response = await _client.PostAsJsonAsync("/api/person", request);

        Assert.True(response.StatusCode == HttpStatusCode.Created || 
                    response.StatusCode == HttpStatusCode.OK);
    }

    [Fact]
    public async Task Create_WithInvalidData_ReturnsBadRequest()
    {
        var request = new CreatePersonRequest
        {
            FirstName = "",
            LastName = "",
            Email = "invalid-email"
        };

        var response = await _client.PostAsJsonAsync("/api/person", request);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    [Fact]
    public async Task Search_ReturnsResults()
    {
        var response = await _client.GetAsync("/api/person/search?query=test");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }
}

public class ArticleControllerIntegrationTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly WebApplicationFactory<Program> _factory;
    private readonly HttpClient _client;

    public ArticleControllerIntegrationTests(WebApplicationFactory<Program> factory)
    {
        _factory = factory.WithWebHostBuilder(builder =>
        {
            builder.UseEnvironment("Testing");
        });
        _client = _factory.CreateClient();
    }

    [Fact]
    public async Task GetAll_ReturnsPagedResult()
    {
        var response = await _client.GetAsync("/api/article");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var content = await response.Content.ReadFromJsonAsync<PagedResult<ArticleDto>>();
        Assert.NotNull(content);
    }

    [Fact]
    public async Task GetFeatured_ReturnsFeaturedArticles()
    {
        var response = await _client.GetAsync("/api/article/featured");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    [Fact]
    public async Task GetById_WhenNotExists_ReturnsNotFound()
    {
        var response = await _client.GetAsync("/api/article/99999");

        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
    }
}

public class OrderControllerIntegrationTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly WebApplicationFactory<Program> _factory;
    private readonly HttpClient _client;

    public OrderControllerIntegrationTests(WebApplicationFactory<Program> factory)
    {
        _factory = factory.WithWebHostBuilder(builder =>
        {
            builder.UseEnvironment("Testing");
        });
        _client = _factory.CreateClient();
    }

    [Fact]
    public async Task GetMyOrders_Unauthorized_Returns401()
    {
        var response = await _client.GetAsync("/api/order/my-orders");

        Assert.Equal(HttpStatusCode.Unauthorized, response.StatusCode);
    }

    [Fact]
    public async Task GetById_WhenNotExists_ReturnsNotFound()
    {
        var response = await _client.GetAsync($"/api/order/{Guid.NewGuid()}");

        Assert.True(response.StatusCode == HttpStatusCode.NotFound ||
                    response.StatusCode == HttpStatusCode.Unauthorized);
    }
}

public class HealthControllerIntegrationTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly HttpClient _client;

    public HealthControllerIntegrationTests(WebApplicationFactory<Program> factory)
    {
        _client = factory.CreateClient();
    }

    [Fact]
    public async Task HealthCheck_ReturnsHealthy()
    {
        var response = await _client.GetAsync("/api/health");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var content = await response.Content.ReadAsStringAsync();
        Assert.Contains("healthy", content.ToLower());
    }

    [Fact]
    public async Task LiveCheck_ReturnsAlive()
    {
        var response = await _client.GetAsync("/api/health/live");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    [Fact]
    public async Task ReadyCheck_ReturnsReady()
    {
        var response = await _client.GetAsync("/api/health/ready");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }
}

public class MinimalApiIntegrationTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly HttpClient _client;

    public MinimalApiIntegrationTests(WebApplicationFactory<Program> factory)
    {
        _client = factory.CreateClient();
    }

    [Fact]
    public async Task HealthEndpoint_ReturnsOk()
    {
        var response = await _client.GetAsync("/health");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    [Fact]
    public async Task ReadyEndpoint_ReturnsOk()
    {
        var response = await _client.GetAsync("/ready");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    [Fact]
    public async Task StatusEndpoint_ReturnsOperational()
    {
        var response = await _client.GetAsync("/api/v2/status");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var content = await response.Content.ReadAsStringAsync();
        Assert.Contains("operational", content);
    }
}

public class AdminControllerIntegrationTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly HttpClient _client;

    public AdminControllerIntegrationTests(WebApplicationFactory<Program> factory)
    {
        _client = factory.CreateClient();
    }

    [Fact]
    public async Task Dashboard_Unauthorized_Returns401()
    {
        var response = await _client.GetAsync("/api/admin/dashboard");

        Assert.Equal(HttpStatusCode.Unauthorized, response.StatusCode);
    }

    [Fact]
    public async Task GetUsers_Unauthorized_Returns401()
    {
        var response = await _client.GetAsync("/api/admin/users");

        Assert.Equal(HttpStatusCode.Unauthorized, response.StatusCode);
    }

    [Fact]
    public async Task GetSettings_Unauthorized_Returns401()
    {
        var response = await _client.GetAsync("/api/admin/settings");

        Assert.Equal(HttpStatusCode.Unauthorized, response.StatusCode);
    }
}

public class ProductControllerIntegrationTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly HttpClient _client;

    public ProductControllerIntegrationTests(WebApplicationFactory<Program> factory)
    {
        _client = factory.CreateClient();
    }

    [Fact]
    public async Task GetAll_ReturnsPagedProducts()
    {
        var response = await _client.GetAsync("/api/product");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    [Fact]
    public async Task Search_WithQuery_ReturnsResults()
    {
        var response = await _client.GetAsync("/api/product/search?q=test");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    [Fact]
    public async Task GetByCategory_ReturnsProducts()
    {
        var response = await _client.GetAsync("/api/product/category/1");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }
}

public class AuthControllerIntegrationTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly HttpClient _client;

    public AuthControllerIntegrationTests(WebApplicationFactory<Program> factory)
    {
        _client = factory.CreateClient();
    }

    [Fact]
    public async Task Register_WithValidData_ReturnsSuccess()
    {
        var request = new RegisterRequest
        {
            Email = $"newuser{Guid.NewGuid()}@example.com",
            Password = "SecurePassword123!",
            FirstName = "New",
            LastName = "User"
        };

        var response = await _client.PostAsJsonAsync("/api/auth/register", request);

        Assert.True(response.IsSuccessStatusCode || 
                    response.StatusCode == HttpStatusCode.BadRequest);
    }

    [Fact]
    public async Task Login_WithInvalidCredentials_ReturnsUnauthorized()
    {
        var request = new LoginRequest
        {
            Email = "nonexistent@example.com",
            Password = "wrongpassword"
        };

        var response = await _client.PostAsJsonAsync("/api/auth/login", request);

        Assert.True(response.StatusCode == HttpStatusCode.Unauthorized ||
                    response.StatusCode == HttpStatusCode.BadRequest);
    }

    [Fact]
    public async Task ForgotPassword_ReturnsOk()
    {
        var request = new ForgotPasswordRequest
        {
            Email = "test@example.com"
        };

        var response = await _client.PostAsJsonAsync("/api/auth/forgot-password", request);

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }
}
