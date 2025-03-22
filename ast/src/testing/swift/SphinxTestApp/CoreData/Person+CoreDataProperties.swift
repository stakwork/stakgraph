//
//  Person+CoreDataProperties.swift
//  
//
//  Created by Tomas Timinskas on 17/03/2025.
//
//

import Foundation
import CoreData


extension Person {

    @nonobjc public class func fetchRequest() -> NSFetchRequest<Person> {
        return NSFetchRequest<Person>(entityName: "Person")
    }

    @NSManaged public var alias: String?
    @NSManaged public var imageUrl: String?
    @NSManaged public var publicKey: String?
    @NSManaged public var routeHint: String?

}
