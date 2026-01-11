using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace CSharpTestServer.Models;

[Table("Persons")]
public class Person
{
    public int Id { get; set; }

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

    public string? AvatarUrl { get; set; }

    public bool IsActive { get; set; } = true;

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    public DateTime? UpdatedAt { get; set; }

    public int? CountryId { get; set; }

    [ForeignKey("CountryId")]
    public Country? Country { get; set; }

    public ICollection<Article> Articles { get; set; } = new List<Article>();

    public ICollection<Order> Orders { get; set; } = new List<Order>();

    public UserProfile? Profile { get; set; }

    public string FullName => $"{FirstName} {LastName}";

    public int CalculateAge()
    {
        var today = DateTime.Today;
        var age = today.Year - DateOfBirth.Year;
        if (DateOfBirth.Date > today.AddYears(-age)) age--;
        return age;
    }

    public void UpdateEmail(string newEmail)
    {
        Email = newEmail;
        UpdatedAt = DateTime.UtcNow;
    }

    public void Deactivate()
    {
        IsActive = false;
        UpdatedAt = DateTime.UtcNow;
    }
}

public class UserProfile
{
    public int Id { get; set; }

    public int PersonId { get; set; }

    [ForeignKey("PersonId")]
    public Person Person { get; set; } = null!;

    public string? Bio { get; set; }

    public string? Website { get; set; }

    public string? TwitterHandle { get; set; }

    public string? LinkedInUrl { get; set; }

    public string? GithubUsername { get; set; }

    public NotificationPreferences Preferences { get; set; } = new();
}

public class NotificationPreferences
{
    public bool EmailNotifications { get; set; } = true;
    public bool PushNotifications { get; set; } = true;
    public bool MarketingEmails { get; set; } = false;
}
