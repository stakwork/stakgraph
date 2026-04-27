//
//  Person+CoreDataClass.swift
//  
//
//  Created by Tomas Timinskas on 17/03/2025.
//
//

import Foundation
import CoreData
// @ast node: Class "Person"
// @ast edge: Operand -> Function "fetchAllObjects" "Person+CoreDataClass.swift"
// @ast node: DataModel "Person"
// @ast node: Function "fetchAllObjects"
// @ast node: Import "import-imports-srctestingswiftlegacyappsphinxtestappcoredatapersoncoredataclassswift-8"

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
