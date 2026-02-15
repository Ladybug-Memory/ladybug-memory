from memory import LadybugMemory


def main() -> None:
    mem = LadybugMemory("memory.lbdb")

    mem.store(
        "User prefers Python for data analysis", memory_type="preference", importance=8
    )
    mem.store("User is allergic to nuts", memory_type="preference", importance=9)
    mem.store("Fixed bug in login handler", memory_type="work", importance=7)
    mem.store(
        "The python is a large snake found in tropical regions",
        memory_type="fact",
        importance=5,
    )

    print("=== Keyword Search for 'python' ===")
    results = mem.search("python")
    for r in results:
        print(f"  - {r.entry.content} (score: {r.score})")

    print("\n=== Semantic Search for 'snake' ===")
    results = mem.semantic_search("snake", limit=3)
    for r in results:
        print(f"  - {r.entry.content} (score: {r.score:.4f})")

    print("\nRecall all memories:")
    for entry in mem.recall():
        print(
            f"  - [{entry.memory_type}] {entry.content} (importance: {entry.importance})"
        )

    print(f"\nTotal memories: {mem.count()}")


if __name__ == "__main__":
    main()
