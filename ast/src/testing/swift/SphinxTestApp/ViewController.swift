//
//  ViewController.swift
//  SphinxTestApp
//
//  Created by Tomas Timinskas on 17/03/2025.
//

import UIKit
import SwiftyJSON

class ViewController: UIViewController {

    @IBOutlet weak var peopleTableView: UITableView!
    
    var persons: [Person] = []
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        deleteObject()
        configureTableView()
        getPeopleAndSave()
    }
    
    func configureTableView() {
        peopleTableView.delegate = self
        peopleTableView.dataSource = self
    }
    
    func getPeopleAndSave() {
        let appDelegate = UIApplication.shared.delegate as! AppDelegate
        
        API.sharedInstance.getPeopleList { (success, people) in
            if success {
                for person in people?.arrayValue ?? [] {
                    let personJson = JSON(person)
                    let person = Person(context: appDelegate.persistentContainer.viewContext)
                    person.alias = personJson["unique_name"].string ?? ""
                    person.imageUrl = personJson["img"].string ?? ""
                    person.publicKey = personJson["owner_pubkey"].string ?? ""
                    person.routeHint = personJson["owner_route_hint"].string ?? ""
                }
                
                appDelegate.saveContext()
            }
            
            self.updateTableView()
        }
    }
    
    func deleteObject() {
        let appDelegate = UIApplication.shared.delegate as! AppDelegate
        let managedContext = appDelegate.persistentContainer.viewContext
        let all = Person.fetchAllObjects(entityName: "Person", context: appDelegate.persistentContainer.viewContext)
        
        managedContext.performAndWait {
            for person in all {
                managedContext.delete(person)
            }
        }
    }
    
    func updateTableView() {
        let appDelegate = UIApplication.shared.delegate as! AppDelegate
        persons = Person.fetchAllObjects(entityName: "Person", context: appDelegate.persistentContainer.viewContext)
        peopleTableView.reloadData()
    }
    
    func updateProfile(person: Person) {
        API.sharedInstance.updatePeopleProfileWith(
            alias: person.alias ?? "",
            imageUrl: person.imageUrl ?? "",
            publicKey: person.publicKey ?? "",
            routeHint: person.routeHint ?? "",
            callback: { success in
                print(success)
            }
        )
    }
}

extension ViewController: UITableViewDataSource, UITableViewDelegate {
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return persons.count
    }

    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "PersonTableViewCell", for: indexPath) as! PersonTableViewCell
        let person = persons[indexPath.row]
        cell.nameLabel.text = "\(person.alias ?? "Unknown")"
        return cell
    }
    
    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        let person = persons[indexPath.row]
        updateProfile(person: person)
    }
}

