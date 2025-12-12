# Feature (E2E)
RSpec.describe "Country management", type: :feature do
  let(:admin_user) { create(:person, role: :admin) }
  
  before do
    sign_in(admin_user)
  end
  
  scenario "admin creates a new country successfully" do
    visit "/countries"
    click_link "New Country"
    
    expect(page).to have_current_path("/countries/new")
    
    within("form#new-country") do
      fill_in "Name", with: "United States"
      fill_in "Code", with: "US"
      fill_in "Population", with: "331000000"
      check "Active"
      click_button "Create Country"
    end
    
    expect(page).to have_current_path("/countries")
    expect(page).to have_content("Country created successfully")
    expect(page).to have_content("United States")
    
    within(".countries-table") do
      expect(page).to have_selector("tr", text: "United States")
      expect(page).to have_selector("td", text: "US")
    end
  end
  
  scenario "shows validation errors when creating invalid country" do
    visit "/countries/new"
    
    within("form#new-country") do
      fill_in "Name", with: ""
      fill_in "Code", with: ""
      click_button "Create Country"
    end
    
    expect(page).to have_current_path("/countries")
    
    within(".error-messages") do
      expect(page).to have_content("Name can't be blank")
      expect(page).to have_content("Code can't be blank")
    end
    
    expect(page).not_to have_content("Country created successfully")
  end
  
  scenario "admin edits an existing country" do
    country = create(:country, name: "Canada", code: "CA")
    
    visit "/countries"
    
    within(".country-row[data-id='#{country.id}']") do
      click_link "Edit"
    end
    
    expect(page).to have_current_path("/countries/#{country.id}/edit")
    
    within("form#edit-country") do
      expect(page).to have_field("Name", with: "Canada")
      
      fill_in "Name", with: "Canada (Updated)"
      fill_in "Population", with: "38000000"
      click_button "Update Country"
    end
    
    expect(page).to have_current_path("/countries")
    expect(page).to have_content("Country updated successfully")
    expect(page).to have_content("Canada (Updated)")
  end
  
  scenario "prevents invalid updates" do
    country = create(:country, name: "Mexico", code: "MX")
    
    visit "/countries/#{country.id}/edit"
    
    within("form#edit-country") do
      fill_in "Name", with: ""
      click_button "Update Country"
    end
    
    within(".error-messages") do
      expect(page).to have_content("Name can't be blank")
    end
    
    # Verify data unchanged
    country.reload
    expect(country.name).to eq("Mexico")
  end
  
  scenario "admin deletes a country" do
    country = create(:country, name: "Germany", code: "DE")
    
    visit "/countries"
    
    within(".country-row[data-id='#{country.id}']") do
      accept_confirm do
        click_button "Delete"
      end
    end
    
    expect(page).to have_content("Country deleted successfully")
    expect(page).not_to have_content("Germany")
    expect(page).not_to have_selector(".country-row[data-id='#{country.id}']")
  end
  
  scenario "lists all countries with pagination" do
    create_list(:country, 25)
    
    visit "/countries"
    
    within(".countries-table") do
      expect(page).to have_selector("tr.country-row", count: 20)
    end
    
    within(".pagination") do
      expect(page).to have_link("2")
      click_link "2"
    end
    
    within(".countries-table") do
      expect(page).to have_selector("tr.country-row", count: 5)
    end
  end
  
  scenario "searches for countries by name" do
    create(:country, name: "France", code: "FR")
    create(:country, name: "Finland", code: "FI")
    create(:country, name: "Japan", code: "JP")
    
    visit "/countries"
    
    within(".search-form") do
      fill_in "Search", with: "F"
      click_button "Search"
    end
    
    within(".countries-table") do
      expect(page).to have_content("France")
      expect(page).to have_content("Finland")
      expect(page).not_to have_content("Japan")
    end
  end
  
  scenario "filters countries by active status" do
    active_country = create(:country, name: "Active Country", active: true)
    inactive_country = create(:country, name: "Inactive Country", active: false)
    
    visit "/countries"
    
    within(".filters") do
      select "Active only", from: "Status"
      click_button "Apply Filters"
    end
    
    within(".countries-table") do
      expect(page).to have_content("Active Country")
      expect(page).not_to have_content("Inactive Country")
    end
  end
  
  scenario "displays country details page" do
    country = create(:country, :european, name: "Spain", code: "ES", population: 47000000)
    
    visit "/countries"
    
    within(".country-row[data-id='#{country.id}']") do
      click_link "View Details"
    end
    
    expect(page).to have_current_path("/countries/#{country.id}")
    
    within(".country-details") do
      expect(page).to have_selector("h1", text: "Spain")
      expect(page).to have_content("Code: ES")
      expect(page).to have_content("Population: 47,000,000")
      expect(page).to have_content("Region: Europe")
    end
  end
  
  scenario "handles duplicate country codes gracefully" do
    create(:country, code: "UK", name: "United Kingdom")
    
    visit "/countries/new"
    
    within("form#new-country") do
      fill_in "Name", with: "Another UK"
      fill_in "Code", with: "UK"
      click_button "Create Country"
    end
    
    within(".error-messages") do
      expect(page).to have_content("Code has already been taken")
    end
  end
  
  scenario "bulk actions on multiple countries" do
    country1 = create(:country, name: "Country 1", active: true)
    country2 = create(:country, name: "Country 2", active: true)
    country3 = create(:country, name: "Country 3", active: true)
    
    visit "/countries"
    
    within(".countries-table") do
      check "select_#{country1.id}"
      check "select_#{country2.id}"
    end
    
    within(".bulk-actions") do
      select "Deactivate", from: "Action"
      click_button "Apply to Selected"
    end
    
    expect(page).to have_content("2 countries updated")
    
    [country1, country2].each do |country|
      within(".country-row[data-id='#{country.id}']") do
        expect(page).to have_selector(".status-badge.inactive")
      end
    end
  end
end
