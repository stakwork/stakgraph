class CreatePeople < ActiveRecord::Migration[7.0]
  def change
    create_table :people do |t|
      t.string :name
      t.string :email

      t.timestamps
    end
    add_index :people, :email, unique: true
  end
end