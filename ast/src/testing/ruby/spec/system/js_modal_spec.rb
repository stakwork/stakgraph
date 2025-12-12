RSpec.describe "JS modal", type: :system, js: true do
  let(:user) { create(:person) }
  
  before do
    driven_by(:selenium_chrome_headless)
    sign_in(user)
  end
  
  describe "opening and closing modal" do
    before { visit "/dashboard" }
    
    it "opens modal when clicking trigger button" do
      click_button "Open Modal"
      
      within("#modal-container", wait: 2) do
        expect(page).to have_selector(".modal.is-active")
        expect(page).to have_content("Modal Title")
      end
    end
    
    it "closes modal when clicking close button" do
      click_button "Open Modal"
      
      within("#modal-container", wait: 2) do
        expect(page).to have_selector(".modal.is-active")
        click_button "Close"
      end
      
      expect(page).to have_no_selector(".modal.is-active", wait: 2)
    end
    
    it "closes modal when clicking overlay backdrop" do
      click_button "Open Modal"
      
      within("#modal-container", wait: 2) do
        expect(page).to have_selector(".modal.is-active")
      end
      
      find(".modal-overlay").click
      expect(page).to have_no_selector(".modal.is-active", wait: 2)
    end
    
    it "closes modal with escape key" do
      click_button "Open Modal"
      
      within("#modal-container", wait: 2) do
        expect(page).to have_selector(".modal.is-active")
      end
      
      page.driver.browser.action.send_keys(:escape).perform
      expect(page).to have_no_selector(".modal.is-active", wait: 2)
    end
  end
  
  describe "modal with form submission" do
    before { visit "/articles" }
    
    it "submits form and closes modal on success" do
      click_button "New Article"
      
      within(".modal-content", wait: 2) do
        expect(page).to have_selector("#article-form")
        
        fill_in "Title", with: "New Article Title"
        fill_in "Body", with: "Article content here"
        click_button "Create Article"
      end
      
      expect(page).to have_no_selector(".modal.is-active", wait: 2)
      expect(page).to have_content("Article created successfully")
      expect(page).to have_content("New Article Title")
    end
    
    it "shows validation errors without closing modal" do
      click_button "New Article"
      
      within(".modal-content", wait: 2) do
        fill_in "Title", with: ""
        click_button "Create Article"
        
        expect(page).to have_selector(".error-message", text: "Title can't be blank", wait: 2)
        expect(page).to have_selector(".modal.is-active")
      end
    end
  end
  
  describe "modal with dynamic content loading" do
    before { visit "/people" }
    
    it "loads content asynchronously when modal opens" do
      first(".person-row").click_link "View Details"
      
      within(".modal-content", wait: 2) do
        expect(page).to have_selector(".loading-spinner")
        expect(page).to have_content("Loading...", wait: 1)
        
        expect(page).to have_no_selector(".loading-spinner", wait: 3)
        expect(page).to have_selector(".person-details")
        expect(page).to have_content(user.name)
      end
    end
    
    it "displays error message if content fails to load" do
      # Simulate network error by visiting non-existent resource
      visit "/people"
      click_link "View Broken Details"
      
      within(".modal-content", wait: 2) do
        expect(page).to have_selector(".error-message", text: "Failed to load content", wait: 3)
        expect(page).to have_button("Retry")
      end
    end
  end
  
  describe "nested modal behavior" do
    before { visit "/settings" }
    
    it "opens second modal on top of first modal" do
      click_button "Open Settings"
      
      within(".modal.settings-modal", wait: 2) do
        expect(page).to have_selector(".modal.is-active")
        click_button "Advanced Options"
      end
      
      within(".modal.advanced-modal", wait: 2) do
        expect(page).to have_selector(".modal.is-active")
        expect(page).to have_content("Advanced Settings")
      end
      
      # Both modals should be active
      expect(page).to have_selector(".modal.is-active", count: 2)
    end
    
    it "closes nested modal without closing parent" do
      click_button "Open Settings"
      
      within(".modal.settings-modal", wait: 2) do
        click_button "Advanced Options"
      end
      
      within(".modal.advanced-modal", wait: 2) do
        click_button "Close"
      end
      
      expect(page).to have_selector(".modal.settings-modal.is-active", wait: 2)
      expect(page).to have_no_selector(".modal.advanced-modal", wait: 2)
    end
  end
  
  describe "modal with confirmation dialog" do
    let!(:article) { create(:article, :published, person: user) }
    
    before { visit "/articles" }
    
    it "shows confirmation modal before destructive action" do
      within(".article-row[data-id='#{article.id}']") do
        click_button "Delete"
      end
      
      within(".modal.confirm-modal", wait: 2) do
        expect(page).to have_content("Are you sure?")
        expect(page).to have_content("This action cannot be undone")
        expect(page).to have_button("Cancel")
        expect(page).to have_button("Delete", class: "danger")
      end
    end
    
    it "cancels action when clicking cancel" do
      within(".article-row[data-id='#{article.id}']") do
        click_button "Delete"
      end
      
      within(".modal.confirm-modal", wait: 2) do
        click_button "Cancel"
      end
      
      expect(page).to have_no_selector(".modal.is-active", wait: 2)
      expect(page).to have_selector(".article-row[data-id='#{article.id}']")
    end
    
    it "proceeds with action when confirming" do
      within(".article-row[data-id='#{article.id}']") do
        click_button "Delete"
      end
      
      within(".modal.confirm-modal", wait: 2) do
        click_button "Delete"
      end
      
      expect(page).to have_no_selector(".modal.is-active", wait: 2)
      expect(page).to have_content("Article deleted successfully")
      expect(page).to have_no_selector(".article-row[data-id='#{article.id}']")
    end
  end
  
  describe "modal accessibility" do
    before { visit "/dashboard" }
    
    it "traps focus within modal when open" do
      click_button "Open Modal"
      
      within(".modal-content", wait: 2) do
        first_focusable = find(".modal-close", visible: :all)
        last_focusable = find("button.submit", visible: :all)
        
        last_focusable.send_keys(:tab)
        expect(page).to have_selector(":focus", match: :first)
      end
    end
    
    it "restores focus to trigger element when closed" do
      trigger_button = find_button("Open Modal")
      trigger_button.click
      
      within(".modal-content", wait: 2) do
        click_button "Close"
      end
      
      expect(page).to have_no_selector(".modal.is-active", wait: 2)
      expect(trigger_button).to match_selector(":focus")
    end
  end
end
