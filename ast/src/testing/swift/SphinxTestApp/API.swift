//
//  API.swift
//  SphinxTestApp
//
//  Created by Tomas Timinskas on 17/03/2025.
//

import Alamofire
import SwiftyJSON

class API {
    class var sharedInstance : API {
        struct Static {
            static let instance = API()
        }
        return Static.instance
    }
    
    public static let host = "localhost:3000"
    
    typealias GetPeopleListCallback = ((Bool, JSON?) -> ())
    typealias CreatePeopleProfile = (Bool) -> ()
    
    func createRequest(
        _ url: String,
        bodyParams: NSDictionary?,
        headers: [String: String] = .init(),
        method: String,
        contentType: String = "application/json",
        token: String? = nil
    ) -> URLRequest? {
        if let nsURL = URL(string: url) {
            var request = URLRequest(url: nsURL)
            request.httpMethod = method
            request.setValue(contentType, forHTTPHeaderField: "Content-Type")

            if let token = token {
                request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            }
            
            for (key, value) in headers {
                request.setValue(value, forHTTPHeaderField: key)
            }

            if let p = bodyParams {
                do {
                    try request.httpBody = JSONSerialization.data(withJSONObject: p, options: [])
                } catch let error as NSError {
                    print("Error: " + error.localizedDescription)
                }
            }

            return request
        } else {
            return nil
        }
    }
    
    public func getPeopleList(
        callback: @escaping GetPeopleListCallback
    ) {
        
        let url = "\(API.host)/people"
        
        guard let request = createRequest(
            url,
            bodyParams: nil,
            method: "GET"
        ) else {
            callback(false, nil)
            return
        }
        
        AF.request(request).responseJSON { (response) in
            switch response.result {
            case .success(let data):
                callback(true, JSON(data))
            case .failure(let error):
                callback(false, nil)
            }
        }
    }
    
    public func updatePeopleProfileWith(
        alias: String,
        imageUrl: String?,
        publicKey: String,
        routeHint: String,
        callback: @escaping CreatePeopleProfile
    ) {
        let url = "\(API.host)/person"
        
        let params: [String: AnyObject] = [
            "owner_pubkey": publicKey as AnyObject,
            "owner_alias": alias as AnyObject,
            "owner_route_hint": routeHint as AnyObject,
            "img": imageUrl as AnyObject
        ]
        
        guard let request = createRequest(
            url,
            bodyParams: params as NSDictionary,
            method: "POST"
        ) else {
            callback(false)
            return
        }
        
        AF.request(request).responseJSON { (response) in
            switch response.result {
            case .success(let data):
                let jsonProfile = JSON(data)
                if let pubKey = jsonProfile["owner_pubkey"].string, pubKey == publicKey {
                    callback(true)
                } else {
                    callback(false)
                }
            case .failure(let error):
                callback(false)
            }
        }
    }
}
