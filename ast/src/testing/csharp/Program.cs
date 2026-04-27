// @ast node: Class "EchoRequest"
// @ast node: Class "Program"
// @ast node: Class "WebhookRequest"
// @ast node: Function "MAPGET__health_closure_L84"
// @ast node: Function "MAPGET__ready_closure_L85"
// @ast node: Function "MAPGET__api_v2_status_closure_L87"
// @ast node: Function "MAPPOST__api_v2_webhook_closure_L88"
// @ast node: Function "MAPGET__ping_closure_L91"
// @ast node: Function "MAPPOST__echo_closure_L92"
// @ast node: Endpoint "/api/v2/status"
// @ast edge: Handler -> Function "MAPGET__api_v2_status_closure_L87" "Program.cs"
// @ast node: Endpoint "/api/v2/webhook"
// @ast edge: Handler -> Function "MAPPOST__api_v2_webhook_closure_L88" "Program.cs"
// @ast node: Endpoint "/echo"
// @ast edge: Handler -> Function "MAPPOST__echo_closure_L92" "Program.cs"
// @ast node: Endpoint "/health"
// @ast edge: Handler -> Function "MAPGET__health_closure_L84" "Program.cs"
// @ast node: Endpoint "/ping"
// @ast edge: Handler -> Function "MAPGET__ping_closure_L91" "Program.cs"
// @ast node: Endpoint "/ready"
// @ast edge: Handler -> Function "MAPGET__ready_closure_L85" "Program.cs"
// @ast node: Import "import-imports-srctestingcsharpprogramcs-22"
using Microsoft.EntityFrameworkCore;
using CSharpTestServer.Data;
using CSharpTestServer.Services;
using CSharpTestServer.Repositories;
using CSharpTestServer.Middleware;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

builder.Services.AddDbContext<ApplicationDbContext>(options =>
    options.UseSqlServer(builder.Configuration.GetConnectionString("DefaultConnection")));

builder.Services.AddScoped<IPersonService, PersonService>();
builder.Services.AddScoped<IArticleService, ArticleService>();
builder.Services.AddScoped<IOrderService, OrderService>();
builder.Services.AddScoped<INotificationService, NotificationService>();
builder.Services.AddScoped<IEmailService, EmailService>();
builder.Services.AddScoped<IPaymentService, PaymentService>();

builder.Services.AddScoped<IPersonRepository, PersonRepository>();
builder.Services.AddScoped<IArticleRepository, ArticleRepository>();
builder.Services.AddScoped<IOrderRepository, OrderRepository>();
builder.Services.AddScoped<IProductRepository, ProductRepository>();

builder.Services.AddScoped<ICacheService, RedisCacheService>();
builder.Services.AddScoped<IFileStorageService, S3FileStorageService>();

builder.Services.AddAutoMapper(typeof(Program));
builder.Services.AddMediatR(cfg => cfg.RegisterServicesFromAssembly(typeof(Program).Assembly));

builder.Services.AddAuthentication("Bearer")
    .AddJwtBearer("Bearer", options =>
    {
        options.Authority = builder.Configuration["Auth:Authority"];
        options.TokenValidationParameters.ValidateAudience = false;
    });

builder.Services.AddAuthorization(options =>
{
    options.AddPolicy("AdminOnly", policy => policy.RequireRole("Admin"));
    options.AddPolicy("UserOrAdmin", policy => policy.RequireRole("User", "Admin"));
});

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();
app.UseMiddleware<RequestLoggingMiddleware>();
app.UseMiddleware<ExceptionHandlingMiddleware>();
app.UseAuthentication();
app.UseAuthorization();

app.MapControllers();

app.MapGet("/health", () => Results.Ok(new { Status = "healthy", Timestamp = DateTime.UtcNow }));
app.MapGet("/ready", () => Results.Ok(new { Status = "ready" }));

app.MapGet("/api/v2/status", () => Results.Ok("operational"));
app.MapPost("/api/v2/webhook", (WebhookRequest request) => Results.Accepted());

app.MapGroup("/api/v2/quick")
    .MapGet("/ping", () => "pong")
    .MapPost("/echo", (EchoRequest req) => Results.Ok(req));

app.Run();

public record WebhookRequest(string EventType, string Payload);
public record EchoRequest(string Message);
public partial class Program { }
