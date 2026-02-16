//
//  Person+CoreDataClass.swift
//  
//
//  Created by Tomas Timinskas on 17/03/2025.
//
//

import Foundation
import CoreData

@objc(Person)
public class Person: NSManagedObject {
    public static func fetchAllObjects<T: NSManagedObject>(entityName: String, context: NSManagedObjectContext) -> [T] {
        let fetchRequest = NSFetchRequest<T>(entityName: entityName)
        
        do {
            let results = try context.fetch(fetchRequest)
            return results
        } catch {
            print("Failed to fetch \(entityName): \(error)")
            return []
        }
    }
}
