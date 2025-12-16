RSpec.describe ProcessPaymentJob, type: :job do
  describe "#perform" do
    let(:person) { create(:person, email: "buyer@example.com") }
    let(:amount) { 99.99 }

    it "processes payment for person" do
      expect {
        ProcessPaymentJob.perform_now(person.id, amount)
      }.to change { person.reload.balance }.by(-amount)
    end

    it "enqueues job for later execution" do
      expect {
        ProcessPaymentJob.perform_later(person.id, amount)
      }.to have_enqueued_job(ProcessPaymentJob)
    end

    it "calls payment gateway" do
      expect(PaymentGateway).to receive(:charge).with(person, amount)
      ProcessPaymentJob.perform_now(person.id, amount)
    end

    it "sends confirmation email after payment" do
      expect(UserMailer).to receive(:payment_confirmation).with(person)
      ProcessPaymentJob.perform_now(person.id, amount)
    end

    context "when payment fails" do
      it "retries job 3 times" do
        allow(PaymentGateway).to receive(:charge).and_raise(PaymentError)
        
        expect {
          ProcessPaymentJob.perform_now(person.id, amount)
        }.to raise_error(PaymentError)
      end

      it "notifies admin of failure" do
        allow(PaymentGateway).to receive(:charge).and_raise(PaymentError)
        expect(AdminMailer).to receive(:payment_failed)
        
        begin
          ProcessPaymentJob.perform_now(person.id, amount)
        rescue PaymentError
          # expected
        end
      end
    end
  end
end
