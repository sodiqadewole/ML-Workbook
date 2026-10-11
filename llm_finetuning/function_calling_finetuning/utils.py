

def preview_dataset_sample(dataset: Dataset, index: int=0) -> None:
    '''Display a formatted preview of a dataset sample inspection'''
    if index >= len(dataset):
        print(f"index {index} is out of range.")
        return
    sample = dataset[index]

    print(f"Dataset sample preview: index: {index}")
    print("=" * 80)

    print("\n Raw Components:")
    print(f"Query: {sample['query'][:200]}{'...' if len(sample['query']) > 200 else ''}")
    print(f"Tools: {sample['tools'][:200]}{'...' if len(sample['tools']) > 200 else ''}")
    print(f"Answers: {sample['answers'][:200]}{'...' if len(sample['answers']) > 200 else ''}")
    
    print(f"\n📝 Complete Training Text:")
    print("-" * 40)
    print(sample['text'])
    print("-" * 40)
    
    print(f"\n📊 Sample Statistics:")
    print(f"   • Text length: {len(sample['text']):,} characters")
    print(f"   • Estimated tokens: ~{len(sample['text']) // 4:,} tokens")
    
    print("\n✅ Preview complete!")