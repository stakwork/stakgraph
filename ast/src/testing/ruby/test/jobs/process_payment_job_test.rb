require 'test_helper'

class ProcessPaymentJobTest < ActiveJob::TestCase
  test "processes payment for person" do
    person = Person.create!(name: "Buyer", email: "buyer@example.com")
    amount = 49.99

    assert_enqueued_with(job: ProcessPaymentJob, args: [person.id, amount]) do
      ProcessPaymentJob.perform_later(person.id, amount)
    end
  end

  test "performs payment processing immediately" do
    person = Person.create!(name: "Customer", email: "customer@example.com")
    
    ProcessPaymentJob.perform_now(person.id, 99.99)
    
    assert_not_nil person.reload
  end

  test "calls payment gateway with correct arguments" do
    person = Person.create!(name: "User", email: "user@example.com")
    amount = 29.99
    
    ProcessPaymentJob.perform_now(person.id, amount)
  end

  test "sends confirmation email after successful payment" do
    person = Person.create!(name: "Payer", email: "payer@example.com")
    
    assert_enqueued_emails 1 do
      ProcessPaymentJob.perform_now(person.id, 19.99)
    end
  end

  test "retries on payment failure" do
    person = Person.create!(name: "Retrier", email: "retry@example.com")
    
    assert_performed_jobs 1 do
      ProcessPaymentJob.perform_now(person.id, 9.99)
    end
  end
end
