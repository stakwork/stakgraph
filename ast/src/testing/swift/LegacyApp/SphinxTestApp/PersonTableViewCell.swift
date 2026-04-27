//
//  PersonTableViewCell.swift
//  SphinxTestApp
//
//  Created by Tomas Timinskas on 17/03/2025.
//

import UIKit
// @ast node: Class "PersonTableViewCell"
// @ast edge: Operand -> Function "awakeFromNib" "PersonTableViewCell.swift"
// @ast edge: Operand -> Function "setSelected" "PersonTableViewCell.swift"
// @ast node: Function "awakeFromNib"
// @ast node: Function "setSelected"
// @ast node: Import "import-imports-srctestingswiftlegacyappsphinxtestapppersontableviewcellswift-7"

class PersonTableViewCell: UITableViewCell {

    @IBOutlet weak var nameLabel: UILabel!
    
    override func awakeFromNib() {
        super.awakeFromNib()
        // Initialization code
    }

    override func setSelected(_ selected: Bool, animated: Bool) {
        super.setSelected(selected, animated: animated)

        // Configure the view for the selected state
    }

}
