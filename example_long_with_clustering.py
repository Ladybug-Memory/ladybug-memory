#!/usr/bin/env python3
"""Test dynamic schema discovery with a challenging 2-page document."""

from memory import LadybugMemory, DynamicSchemaDiscovery


def main():
    memory = LadybugMemory(
        db_path="memory.lbdb",
        enable_entity_extraction=True,
        gliner_model="fastino/gliner2-base-v1",
        entity_confidence_threshold=0.80,
    )

    # Two-page document with diverse entities
    content = """
    The Rise of Artificial Intelligence in Healthcare

    In recent years, artificial intelligence has transformed the healthcare industry
    in unprecedented ways. Dr. Sarah Chen, Chief Medical Officer at Stanford Health
    Care in Palo Alto, California, has been at the forefront of this revolution.
    Working alongside researchers from MIT, Harvard Medical School, and Johns Hopkins
    University, her team has developed groundbreaking diagnostic tools using machine
    learning algorithms.

    The project, which began in March 2019, received initial funding of $50 million
    from the National Institutes of Health (NIH) and the Bill & Melinda Gates Foundation.
    Major technology partners include Microsoft, Google DeepMind, and IBM Watson Health.
    The collaboration spans across multiple continents, with research centers in Boston,
    London, Tokyo, and Singapore.

    Dr. Chen's colleague, Professor James Morrison from Oxford University, published
    their findings in Nature Medicine on September 15, 2023. The paper, titled "Deep
    Learning Applications in Radiology," has been cited over 2,000 times and was
    presented at the International Conference on Machine Learning (ICML) in Vienna,
    Austria. Other keynote speakers at the conference included Dr. Yuki Tanaka from
    the University of Tokyo and Dr. Klaus Mueller from the Max Planck Institute in
    Munich, Germany.

    The pharmaceutical industry has taken notice. Pfizer, Moderna, and Roche have
    all announced partnerships with AI startups. Johnson & Johnson invested $200
    million in Tempus, a Chicago-based company founded by Eric Lefkofsky. Meanwhile,
    Novartis partnered with BenevolentAI in London, and AstraZeneca collaborated
    with insitro in San Francisco.

    Regulatory agencies are racing to keep pace. The FDA in the United States
    approved 91 AI-enabled medical devices in 2023, up from 50 in 2022. Dr. Jennifer
    Rodriguez, who leads the FDA's digital health division, testified before Congress
    on March 8, 2024, about the need for updated regulations. Similar discussions
    are happening at the European Medicines Agency (EMA) in Amsterdam and Japan's
    Pharmaceuticals and Medical Devices Agency (PMDA) in Tokyo.

    Patients are seeing real benefits. At the Mayo Clinic in Rochester, Minnesota,
    AI systems have reduced diagnosis time for rare diseases by 40%. The Cleveland
    Clinic in Ohio reported similar improvements. Kaiser Permanente, operating across
    California, has deployed AI chatbots that handle over 1 million patient queries
    monthly. NHS England, led by Dr. Emma Thompson, is implementing similar systems
    across hospitals in Manchester, Birmingham, and Leeds.

    Ethical concerns remain prominent. Dr. Timnit Gebru, founder of the DAIR Institute
    in Washington D.C., has raised concerns about bias in medical AI systems. Her
    research, published in JAMA on February 28, 2024, showed disparities in diagnostic
    accuracy across different demographic groups. Senator Elizabeth Warren and
    Representative Ro Khanna have introduced legislation to address these issues,
    with hearings scheduled for April 15, 2024, on Capitol Hill.

    International cooperation is growing. The World Health Organization (WHO),
    headquartered in Geneva, Switzerland, established the AI in Health Initiative
    in January 2023. Director-General Dr. Tedros Adhanom Ghebreyesus appointed
    Dr. Maria Santos from Brazil and Dr. Wei Zhang from China to lead the effort.
    The initiative involves partnerships with governments in India, South Africa,
    Australia, and Canada.

    Looking ahead, the field shows no signs of slowing down. NVIDIA announced the
    Clara platform for healthcare AI at their GTC conference in San Jose on
    March 18, 2024. CEO Jensen Huang demonstrated applications in drug discovery,
    medical imaging, and genomics. Competitors AMD and Intel are developing similar
    solutions. Apple, through their health division led by Dr. Sumbul Desai, is
    exploring AI integration with the Apple Watch and HealthKit platform.

    Academic institutions continue to push boundaries. Carnegie Mellon University
    in Pittsburgh received a $100 million grant from the Simons Foundation to
    establish the AI for Health Institute. The University of Toronto, where
    Geoffrey Hinton pioneered deep learning, launched a new master's program in
    health AI in September 2023. ETH Zurich in Switzerland and Tsinghua University
    in Beijing have announced a joint research initiative with initial funding of
    €30 million.

    The economic impact is substantial. According to a report by McKinsey & Company
    published on January 10, 2024, AI in healthcare could generate $150 billion in
    annual savings by 2030. Goldman Sachs and Morgan Stanley have increased their
    investment recommendations for the sector. Venture capital firms including
    Sequoia Capital, Andreessen Horowitz, and Khosla Ventures have collectively
    invested over $5 billion in health AI startups since 2020.

    Startups are flourishing. PathAI in Boston, founded by Dr. Andy Beck, raised
    $165 million in Series C funding in October 2023. Viz.ai in San Francisco,
    led by Dr. Chris Mansi, received FDA approval for their stroke detection
    algorithm. Paige AI in New York became the first company to receive FDA
    approval for an AI-based cancer diagnostic tool. Other notable startups include
    Olive AI in Columbus, Ohio, and Babylon Health in London.

    As we move into 2024, the integration of AI in healthcare continues to evolve.
    The next generation of medical professionals is being trained differently.
    At the Perelman School of Medicine at the University of Pennsylvania, Dean
    Dr. J. Larry Jameson has incorporated AI literacy into the curriculum. Similar
    changes are happening at UCSF, Duke University School of Medicine, and the
    David Geffen School of Medicine at UCLA. The future of medicine is being
    written not just in laboratories and hospitals, but in the algorithms that
    increasingly power them.
    """

    print("=" * 60)
    print("Two-Page Document Knowledge Graph Extraction")
    print("=" * 60)
    print(f"\nDocument length: {len(content)} characters")

    # Store with entity extraction
    print("\nExtracting entities...")
    entry, entities = memory.store_with_entities(
        content=content,
        memory_type="article",
        importance=9,
        extract_entities=True,
    )

    print(f"Total entities extracted: {len(entities)}")

    # Group by type
    type_counts = {}
    for entity in entities:
        etype = entity.entity_type
        type_counts[etype] = type_counts.get(etype, 0) + 1

    print("\nEntities by type:")
    for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {etype}: {count}")
        # Show sample entities
        samples = [e.text for e in entities if e.entity_type == etype][:5]
        for s in samples:
            print(f"    - {s}")

    # Run schema discovery with LLM-based type naming
    print("\nRunning dynamic schema discovery with LLM naming...")

    schema_discovery = DynamicSchemaDiscovery(
        similarity_threshold=0.70,
        min_cluster_size=2,
        min_confidence=0.50,
        llm_model="ollama/granite4:tiny-h",  # Use local Ollama with granite4:tiny-h
    )

    discovered_schemas, entity_to_type = schema_discovery.discover_schema(entities)

    print(f"\nDiscovered {len(discovered_schemas)} entity clusters:")
    for schema in discovered_schemas:
        print(
            f"\n  {schema.type_name.upper()} (confidence: {schema.confidence:.2f}, size: {schema.size})"
        )
        print(f"  Samples: {', '.join(schema.sample_entities[:6])}")

    # Create tables
    schema_dicts = [
        {
            "type_name": s.type_name,
            "confidence": s.confidence,
            "sample_entities": s.sample_entities,
            "cluster_id": s.cluster_id,
            "size": s.size,
        }
        for s in discovered_schemas
    ]

    table_mapping = memory.create_dynamic_schema_tables(schema_dicts)

    print(f"\nCreated {len(table_mapping)} dynamic tables:")
    for type_name, table_name in table_mapping.items():
        print(f"  - {type_name} -> {table_name}")

    print("\n" + "=" * 60)
    print("Knowledge graph extraction completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
