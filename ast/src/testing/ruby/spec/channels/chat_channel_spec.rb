RSpec.describe ChatChannel, type: :channel do
  let(:person) { create(:person, name: "Alice") }

  before do
    stub_connection(current_person: person)
  end

  describe "#subscribed" do
    it "successfully subscribes to chat stream" do
      subscribe(room: "general")
      
      expect(subscription).to be_confirmed
      expect(subscription).to have_stream_from("chat_general")
    end

    it "subscribes to person-specific stream" do
      subscribe(room: person.id)
      
      expect(subscription).to be_confirmed
      expect(subscription).to have_stream_for(person)
    end

    it "rejects subscription without room" do
      subscribe
      
      expect(subscription).to be_rejected
    end
  end

  describe "#unsubscribed" do
    it "stops all streams when unsubscribed" do
      subscribe(room: "general")
      unsubscribe
      
      expect(subscription).not_to have_streams
    end
  end

  describe "#send_message" do
    before do
      subscribe(room: "general")
    end

    it "broadcasts message to room" do
      expect {
        perform :send_message, message: "Hello World"
      }.to have_broadcasted_to("chat_general").with(
        message: "Hello World",
        sender: person.name
      )
    end

    it "includes timestamp in broadcast" do
      expect {
        perform :send_message, message: "Test"
      }.to have_broadcasted_to("chat_general").with(hash_including(:timestamp))
    end

    it "rejects empty messages" do
      expect {
        perform :send_message, message: ""
      }.not_to have_broadcasted_to("chat_general")
    end
  end

  describe "#typing" do
    before do
      subscribe(room: "general")
    end

    it "broadcasts typing indicator" do
      expect {
        perform :typing, typing: true
      }.to have_broadcasted_to("chat_general").with(
        typing: true,
        person_id: person.id
      )
    end

    it "broadcasts stop typing" do
      expect {
        perform :typing, typing: false
      }.to have_broadcasted_to("chat_general").with(
        typing: false,
        person_id: person.id
      )
    end
  end
end
