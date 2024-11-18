import streamlit as st
from Bio.Seq import Seq
from Bio.Data import CodonTable
import numpy as np
import pandas as pd
import openai

# OpenAI API Key (from a secret file)
openai.api_key = "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" 

def bioinformatics_chatbot(prompt):
    """
    Function to interact with OpenAI GPT-4 model for bioinformatics queries.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",  
            messages=[
                {"role": "system", "content": "You are a bioinformatics expert."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"


# Function to find Open Reading Frames (ORFs) in a DNA sequence
def find_orfs_ncbi(sequence, min_length, genetic_code, start_codon_policy, ignore_nested=False):
    # Parameters:
    # sequence: str - input DNA sequence
    # min_length: int - minimum ORF length
    # genetic_code: int - NCBI translation table ID
    # start_codon_policy: str - start codon policy
    # ignore_nested: bool - whether to ignore nested ORFs
    codon_table = CodonTable.unambiguous_dna_by_id[genetic_code]
    
    # Define start codons based on user policy
    if start_codon_policy == "ATG only":
        start_codons = ["ATG"]
    elif start_codon_policy == "ATG and alternative initiation codons":
        start_codons = codon_table.start_codons
    else:  # Any sense codon
        start_codons = [codon for codon in codon_table.forward_table.keys()]
    
    stop_codons = set(codon_table.stop_codons)
    sequence_length = len(sequence)
    
    # Helper function to find ORFs in a given frame
    def find_in_frame(sequence, frame, strand):
        orfs = []
        start = None
        # Iterate over the sequence in the given frame
        for i in range(frame, len(sequence), 3):
            codon = sequence[i:i+3]
            if len(codon) < 3:
                break
            
            if codon in start_codons and start is None:
                start = i
            
            if codon in stop_codons and start is not None:
                end = i + 3
                orf_length = end - start
                if orf_length >= min_length:
                    orfs.append({
                        "Strand": strand,
                        "Frame": frame + 1 if strand == "Forward" else -(frame + 1),
                        "Start": start + 1 if strand == "Forward" else sequence_length - end + 1,
                        "End": end if strand == "Forward" else sequence_length - start,
                        "Length (nt)": orf_length,
                        "Protein Length (aa)": orf_length // 3,
                        "Sequence": sequence[start:end],
                        "Translated Protein": Seq(sequence[start:end]).translate(table=genetic_code)
                    })
                start = None
        
        return orfs
    
    orfs = []
    for frame in range(3):
        orfs.extend(find_in_frame(sequence, frame, "Forward"))
    
    reverse_seq = str(Seq(sequence).reverse_complement())
    for frame in range(3):
        orfs.extend(find_in_frame(reverse_seq, frame, "Reverse"))
    
    if ignore_nested:
        orfs = filter_nested_orfs(orfs)
    
    return orfs
# Helper function to filter out nested ORFs
def filter_nested_orfs(orfs):
    # Sort ORFs by start position and then by end position
    orfs_sorted = sorted(orfs, key=lambda x: (x["Start"], -x["End"])) 
    non_nested = []
    # Iterate through sorted ORFs and keep the ones that are not nested
    for orf in orfs_sorted:
        if not any(orf["Start"] >= prev["Start"] and orf["End"] <= prev["End"] for prev in non_nested):
            non_nested.append(orf)
    return non_nested

# Function to perform global sequence alignment using the Needleman-Wunsch algorithm
def needleman_wunsch(seq1, seq2, match_score=1, mismatch_score=-1, gap_penalty=-2):
    # Parameters:
    # seq1: str - first input sequence
    # seq2: str - second input sequence
    # match_score: int - score for a match
    # mismatch_score: int - score for a mismatch
    # gap_penalty: int - penalty for a gap

    # Initialize the scoring matrix
    n = len(seq1) + 1
    m = len(seq2) + 1
    score_matrix = np.zeros((n, m), dtype=int)

    # Fill the first row and column with gap penalties
    for i in range(1, n):
        score_matrix[i][0] = score_matrix[i-1][0] + gap_penalty
    for j in range(1, m):
        score_matrix[0][j] = score_matrix[0][j-1] + gap_penalty

    # Fill the scoring matrix based on the match/mismatch and gap penalties
    for i in range(1, n):
        for j in range(1, m):
            if seq1[i-1] == seq2[j-1]:
                score = match_score
            else:
                score = mismatch_score
            match = score_matrix[i-1][j-1] + score
            delete = score_matrix[i-1][j] + gap_penalty
            insert = score_matrix[i][j-1] + gap_penalty
            score_matrix[i][j] = max(match, delete, insert)

    # Traceback to find the optimal alignment
    align1 = ""
    align2 = ""
    i, j = len(seq1), len(seq2)

    while i > 0 and j > 0:
        score_current = score_matrix[i][j]
        score_diagonal = score_matrix[i-1][j-1]
        score_up = score_matrix[i][j-1]
        score_left = score_matrix[i-1][j]

        # Check the three possible directions
        if score_current == score_diagonal + (match_score if seq1[i-1] == seq2[j-1] else mismatch_score):
            align1 = seq1[i-1] + align1
            align2 = seq2[j-1] + align2
            i -= 1
            j -= 1
        elif score_current == score_left + gap_penalty:
            align1 = seq1[i-1] + align1
            align2 = "-" + align2
            i -= 1
        elif score_current == score_up + gap_penalty:
            align1 = "-" + align1
            align2 = seq2[j-1] + align2
            j -= 1

    # Add remaining gaps if we are not at the start of seq1 or seq2
    while i > 0:
        align1 = seq1[i-1] + align1
        align2 = "-" + align2
        i -= 1
    while j > 0:
        align1 = "-" + align1
        align2 = seq2[j-1] + align2
        j -= 1

    return align1, align2, score_matrix

# Set the page title and layout
st.set_page_config(page_title="Bioinformatics Tools", layout="wide")

# Sidebar for navigation
st.sidebar.title("Bioinformatics Tools")
selection = st.sidebar.radio("Choose a tool:", ["Home", "DNA Translate/Transcribe", "Global Sequence Alignment", "ORFs Finder", "AI Bioinformatics Chatbot"])

# Home page
if selection == "Home":
    st.title("Welcome to the Bioinformatics Tools")
    st.write("Use the sidebar to select one of the available bioinformatics tools.")
    st.title("About the Project")
    st.write("This project is a Streamlit web application that provides bioinformatics tools for DNA sequence analysis.")
    st.write("The application is built using Python, Biopython, and Streamlit.")
    st.write("""Creators:
                    \n  Abdullah Mohammed Alameri
                    \n  Hamad Saleh Almarzooqi
                    \n  Mayed Almemari
             """)
    st.write("This project is part of the Bioinformatics course at the United Arab Emirates University (UAEU).")
    
elif selection == "DNA Translate/Transcribe":
    st.title("DNA Translate or Transcribe")

    # User input for DNA sequence
    sequence = st.text_area("Enter DNA Sequence:")
    action = st.selectbox("Choose Action:", ["Transcribe (DNA to RNA)", "Translate (DNA to Protein)"])

    if st.button("Submit"):
        result = ""
        dna_seq = Seq(sequence)
        if action == "Transcribe (DNA to RNA)":
            # DNA Transcription logic using Biopython Seq object + transcribe()
            result = dna_seq.transcribe()
        elif action == "Translate (DNA to Protein)":
            # DNA Translation logic using Biopython Seq object + translate() default to standard genetic code
            result = dna_seq.translate()
        st.write("**Result:**", result)

    # Expander theory and explanation section
    with st.expander("Theory and Explanation"):
        st.subheader("Understanding DNA Translation and Transcription")
        
        st.markdown("""
        **DNA Transcription**:  
        Transcription is the process of converting a DNA template strand into a complementary RNA strand. This step is crucial in gene expression, where the RNA serves as a template for protein synthesis. 

        **DNA Translation**:  
        Translation involves converting the mRNA sequence into a sequence of amino acids to form a protein. This process uses the genetic code, which maps nucleotide triplets (codons) to specific amino acids.

        **Algorithm Used in This Tool**:
        1. **Transcription**:
            - Replace thymine (T) in DNA with uracil (U) to form RNA.
        2. **Translation**:
            - Convert RNA codons into amino acids using a standard genetic code.
        
        **References**:
        St. Clair, C. (2016). *Exploring Bioinformatics* (2nd ed.). Jones & Bartlett Learning.
        """)

# Global Sequence Alignment tool
elif selection == "Global Sequence Alignment":
    st.title("Global Sequence Alignment")

    # User inputs
    sequence1 = st.text_area("Enter Sequence 1:")
    sequence2 = st.text_area("Enter Sequence 2:")
    matching_score = st.number_input("Match score", -100, 100, 1)
    mismatching_score = st.number_input("Mismatch score", -100, 100, -1)
    gap_score = st.number_input("Gap penalty", -100, 100, -2)

    # Perform sequence alignment
    if st.button("Align Sequences"):
        valid_nucleotides = {"A", "C", "G", "U", "T"}
        if all(char in valid_nucleotides for char in sequence1.upper()) and all(char in valid_nucleotides for char in sequence2.upper()):
            # Perform global sequence alignment using Needleman-Wunsch function
            align1, align2, score_table = needleman_wunsch(sequence1, sequence2, matching_score, mismatching_score, gap_score)
            multi_index_columns = pd.MultiIndex.from_tuples([(f"Pos {i}", col) for i, col in enumerate("-" + sequence2.upper())])
            multi_index_index = pd.MultiIndex.from_tuples([(f"Pos {i}", row) for i, row in enumerate("-" + sequence1.upper())])
            alignment_table = pd.DataFrame(score_table, index=multi_index_index, columns=multi_index_columns)
            
            # Display the alignment results
            st.write("Alignment result:")
            st.write(align1)
            st.write(align2)
            st.write("Alignment Score:", score_table[-1][-1])
            st.write("Alignment Matrix:")
            st.dataframe(alignment_table)
        else:
            st.write("Please enter valid DNA sequences.")

    # Expander theory and explanation section
    with st.expander("Theory and Explanation"):
        st.subheader("Understanding Global Sequence Alignment")
        
        st.markdown("""
        **Global Sequence Alignment**:  
        This technique is used to align two sequences from end to end, identifying similarities and differences across their entire length. It is particularly useful for comparing sequences of similar lengths, such as homologous genes from different species.

        **Needleman-Wunsch Algorithm**:
        The Needleman-Wunsch algorithm is a dynamic programming approach to global sequence alignment. It ensures the optimal alignment by considering gaps, matches, and mismatches between sequences.

        **Algorithm Steps**:
        1. **Initialization**: 
           - Construct a matrix where the first row and column are initialized with gap penalties.
        2. **Matrix Filling**:
           - Populate each cell based on the match, mismatch, or gap penalties.
           - Calculate the score for each possible alignment path.
           - Choose the highest score for each cell.
           - Where horizontal or vertical represents a gap, and diagonal movement represents a match or mismatch.
        3. **Traceback**:
           - Backtrack from the bottom-right corner to determine the optimal alignment.
        
        **Practical Use Cases**:
        - Identifying conserved regions between species.
        - Detecting mutations in genetic sequences.
        
        **References**:
        St. Clair, C. (2016). *Exploring Bioinformatics* (2nd ed., pp. 43–45). Jones & Bartlett Learning.
        """)

# ORFs Finder tool
elif selection == "ORFs Finder":
    st.title("ORFs Finder Tool")

    # User inputs
    sequence_input = st.text_area("Enter DNA sequence:").strip().replace("\n", "")
    min_length = st.number_input("Minimum ORF length", min_value=1, value=75)

    # Select genetic code (translation table) using a dropdown
    genetic_code = st.selectbox("Select Genetic Code (Translation Table):",
        [(f"{k}: {v.names[0]}", k) for k, v in CodonTable.unambiguous_dna_by_id.items()], format_func=lambda x: x[0])[1]

    # Select start codon policy using radio buttons    
    start_codon_policy = st.radio("ORF start codon to use:", ["ATG only", "ATG and alternative initiation codons", "Any sense codon"])
    ignore_nested = st.checkbox("Ignore nested ORFs", value=False)

    # Find ORFs in the input sequence
    if st.button("Find ORFs"):
        orfs = find_orfs_ncbi(sequence_input, min_length, genetic_code, start_codon_policy, ignore_nested)
    
        # Convert ORFs list to DataFrame
        orf_df = pd.DataFrame(orfs)
    
        # Convert `Seq` objects to strings in the "Translated Protein" column
        if "Translated Protein" in orf_df.columns:
            orf_df["Translated Protein"] = orf_df["Translated Protein"].astype(str)
    
        # Display the ORFs DataFrame
        st.write(f"Found {len(orf_df)} ORFs:")
        st.dataframe(orf_df)

    # Expander theory and explanation section
    with st.expander("Theory and Explanation"):
        st.subheader("Understanding Open Reading Frames (ORFs)")

        st.markdown("""
        **What is an ORF?**  
        An Open Reading Frame (ORF) is a continuous stretch of DNA that starts with a start codon (e.g., ATG) and ends with a stop codon (e.g., TAA, TAG, TGA). ORFs are critical for gene prediction as they often represent protein-coding regions.

        **Algorithm for ORF Finding**:
        1. **Identify Start and Stop Codons**:
            - Scan the DNA sequence for all occurrences of start and stop codons based on the selected start codon policy.
        2. **Calculate ORF Lengths**:
            - Measure the distance between start and stop codons.
        3. **Filter ORFs by Minimum Length**:
            - Exclude ORFs shorter than the user-defined threshold.
        4. **Handle Nested ORFs** (if selected):
            - Ignore shorter ORFs fully contained within larger ORFs.

        **Practical Applications**:
        - **Gene Prediction**: Identifying regions likely to encode proteins.
        - **Comparative Genomics**: Analyzing genome sequences to discover evolutionary relationships.
        
        **References**:
        St. Clair, C. (2016). *Exploring Bioinformatics* (2nd ed., pp. 175–177). Jones & Bartlett Learning.
        """)

    # Chatbot Interface
elif selection == "AI Bioinformatics Chatbot":
    st.title("Bioinformatics Chatbot")
    st.write("Ask me anything about bioinformatics!")

    user_input = st.text_area("Enter your question:")
    
    if st.button("Ask"):
        if user_input.strip():
            with st.spinner("Thinking..."):
                answer = bioinformatics_chatbot(user_input)
            st.write("**Chatbot Response:**")
            st.write(answer)
        else:
            st.write("Please enter a valid question.")
            
