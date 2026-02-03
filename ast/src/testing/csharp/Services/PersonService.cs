using CSharpTestServer.DTOs;
using CSharpTestServer.Models;
using CSharpTestServer.Repositories;
using AutoMapper;

namespace CSharpTestServer.Services;

public interface IPersonService
{
    Task<IEnumerable<PersonDto>> GetAllAsync(int page, int pageSize);
    Task<PersonDto?> GetByIdAsync(int id);
    Task<IEnumerable<PersonDto>> SearchAsync(string query);
    Task<PersonDto> CreateAsync(CreatePersonRequest request);
    Task<PersonDto?> UpdateAsync(int id, UpdatePersonRequest request);
    Task<PersonDto?> PartialUpdateAsync(int id, PatchPersonRequest request);
    Task<bool> DeleteAsync(int id);
    Task<IEnumerable<ArticleDto>> GetArticlesAsync(int personId);
    Task<string> UploadAvatarAsync(int id, IFormFile file);
}

public class PersonService : IPersonService
{
    private readonly IPersonRepository _personRepository;
    private readonly IArticleRepository _articleRepository;
    private readonly IFileStorageService _fileStorageService;
    private readonly IMapper _mapper;
    private readonly ILogger<PersonService> _logger;

    public PersonService(
        IPersonRepository personRepository,
        IArticleRepository articleRepository,
        IFileStorageService fileStorageService,
        IMapper mapper,
        ILogger<PersonService> logger)
    {
        _personRepository = personRepository;
        _articleRepository = articleRepository;
        _fileStorageService = fileStorageService;
        _mapper = mapper;
        _logger = logger;
    }

    public async Task<IEnumerable<PersonDto>> GetAllAsync(int page, int pageSize)
    {
        var persons = await _personRepository.GetAllAsync(page, pageSize);
        return _mapper.Map<IEnumerable<PersonDto>>(persons);
    }

    public async Task<PersonDto?> GetByIdAsync(int id)
    {
        var person = await _personRepository.GetByIdAsync(id);
        return person != null ? _mapper.Map<PersonDto>(person) : null;
    }

    public async Task<IEnumerable<PersonDto>> SearchAsync(string query)
    {
        var persons = await _personRepository.SearchAsync(query);
        return _mapper.Map<IEnumerable<PersonDto>>(persons);
    }

    public async Task<PersonDto> CreateAsync(CreatePersonRequest request)
    {
        var person = new Person
        {
            FirstName = request.FirstName,
            LastName = request.LastName,
            Email = request.Email,
            Phone = request.Phone,
            DateOfBirth = request.DateOfBirth,
            CountryId = request.CountryId
        };

        await _personRepository.AddAsync(person);
        _logger.LogInformation("Created person with id {Id}", person.Id);

        return _mapper.Map<PersonDto>(person);
    }

    public async Task<PersonDto?> UpdateAsync(int id, UpdatePersonRequest request)
    {
        var person = await _personRepository.GetByIdAsync(id);
        if (person == null)
        {
            return null;
        }

        person.FirstName = request.FirstName;
        person.LastName = request.LastName;
        person.Email = request.Email;
        person.Phone = request.Phone;
        person.DateOfBirth = request.DateOfBirth;
        person.CountryId = request.CountryId;
        person.UpdatedAt = DateTime.UtcNow;

        await _personRepository.UpdateAsync(person);
        return _mapper.Map<PersonDto>(person);
    }

    public async Task<PersonDto?> PartialUpdateAsync(int id, PatchPersonRequest request)
    {
        var person = await _personRepository.GetByIdAsync(id);
        if (person == null)
        {
            return null;
        }

        if (request.FirstName != null) person.FirstName = request.FirstName;
        if (request.LastName != null) person.LastName = request.LastName;
        if (request.Email != null) person.Email = request.Email;
        if (request.Phone != null) person.Phone = request.Phone;
        person.UpdatedAt = DateTime.UtcNow;

        await _personRepository.UpdateAsync(person);
        return _mapper.Map<PersonDto>(person);
    }

    public async Task<bool> DeleteAsync(int id)
    {
        var person = await _personRepository.GetByIdAsync(id);
        if (person == null)
        {
            return false;
        }

        await _personRepository.DeleteAsync(id);
        _logger.LogInformation("Deleted person with id {Id}", id);
        return true;
    }

    public async Task<IEnumerable<ArticleDto>> GetArticlesAsync(int personId)
    {
        var articles = await _articleRepository.GetByAuthorIdAsync(personId);
        return _mapper.Map<IEnumerable<ArticleDto>>(articles);
    }

    public async Task<string> UploadAvatarAsync(int id, IFormFile file)
    {
        var url = await _fileStorageService.UploadAsync(file, $"avatars/{id}");
        
        var person = await _personRepository.GetByIdAsync(id);
        if (person != null)
        {
            person.AvatarUrl = url;
            await _personRepository.UpdateAsync(person);
        }

        return url;
    }
}
