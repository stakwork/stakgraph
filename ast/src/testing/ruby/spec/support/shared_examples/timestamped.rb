RSpec.shared_examples "a timestamped model" do
  it { should respond_to(:created_at) }
  it { should respond_to(:updated_at) }
  
  it "sets created_at on creation" do
    subject.save
    expect(subject.created_at).not_to be_nil
  end
  
  it "updates updated_at on modification" do
    subject.save
    original_updated_at = subject.updated_at
    sleep 0.001
    subject.touch
    expect(subject.updated_at).to be > original_updated_at
  end
end
