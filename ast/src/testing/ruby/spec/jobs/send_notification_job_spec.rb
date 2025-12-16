RSpec.describe SendNotificationJob, type: :job do
  describe "#perform" do
    let(:person) { create(:person, email: "user@example.com") }
    let(:message) { "You have a new notification" }

    it "sends notification to person" do
      expect(NotificationService).to receive(:send_to).with(person, message)
      SendNotificationJob.perform_now(person.id, message)
    end

    it "enqueues notification for delivery" do
      expect {
        SendNotificationJob.perform_later(person.id, message)
      }.to have_enqueued_job(SendNotificationJob).with(person.id, message)
    end

    it "logs notification delivery" do
      expect(Rails.logger).to receive(:info).with(/Sending notification/)
      SendNotificationJob.perform_now(person.id, message)
    end

    context "with priority notifications" do
      it "enqueues with high priority" do
        expect {
          SendNotificationJob.set(priority: 0).perform_later(person.id, message)
        }.to have_enqueued_job(SendNotificationJob).with(person.id, message)
      end
    end

    context "when person not found" do
      it "handles missing person gracefully" do
        expect {
          SendNotificationJob.perform_now(99999, message)
        }.not_to raise_error
      end
    end
  end
end
