import Foundation

enum APIError: Error {
    case invalidURL
    case networkError(Error)
    case decodingError(Error)
}

struct APIClient<T: Codable> {
    var session: URLSession = .shared
    
    func fetch(endpoint: Endpoint) async throws -> T {
        guard let url = endpoint.url else {
            throw APIError.invalidURL
        }
        
        let (data, response) = try await session.data(from: url)
        
        guard let httpResponse = response as? HTTPURLResponse, 
              (200...299).contains(httpResponse.statusCode) else {
            throw APIError.networkError(NSError(domain: "HTTP", code: 500, userInfo: nil))
        }
        
        do {
            let decoder = JSONDecoder()
            return try decoder.decode(T.self, from: data)
        } catch {
            throw APIError.decodingError(error)
        }
    }
    
    func post<U: Encodable>(endpoint: Endpoint, body: U) async throws -> T {
        guard let url = endpoint.url else { throw APIError.invalidURL }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.httpBody = try? JSONEncoder().encode(body)
        
        let (data, _) = try await session.data(for: request)
        return try JSONDecoder().decode(T.self, from: data)
    }
}
