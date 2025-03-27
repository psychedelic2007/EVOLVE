import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os
import argparse

class SequenceSimilarityChecker:
    def __init__(self, reference_sequences=None):
        """
        Initialize the sequence similarity checker with optional reference sequences.
        
        :param reference_sequences: Dict of reference sequences, default is pre-defined viral sequences
        """
        self.default_sequences = {
            'Omicron': 'MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLYSTQDLFLPFFSNVTWFHVISGTNGTKRFDNPVLPFNDGVYFASIEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLDHKNNKSWMESGVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFXEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYRYRLFRKSNLKPFERDISTEIYQAGSKPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLXGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQGVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEXVNNSYECDIPIGAGICASYQTQTXSHRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLXRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKXFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFKGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNXNAQALNTLVKQLSSXFGAISSVLNDIXSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELXSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT',
            'Influenza H1N1': 'MKAKLLVLLYAFVATDADTICIGYHANNSTDTVDTIFEKNVAVTHSVNLLEDRHNGKLCKLKGIAPLQLGKCNITGWLLGNPECDSLLPARSWSYIVETPNSENGACYPGDFIDYEELREQLSSVSSLERFEIFPKESSWPNHTFNGVTVSCSHRGKSSFYRNLLWLTKKGDSYPKLTNSYVNNKGKEVLVLWGVHHPSSSDEQQSLYSNGNAYVSVASSNYNRRFTPEIAARPKVKDQHGRMNYYWTLLEPGDTIIFEATGNLIAPWYAFALSRGFESGIITSNASMHECNTKCQTPQGSINSNLPFQNIHPVTIGECPKYVRSTKLRMVTGLRNIPSIQYRGLFGAIAGFIEGGWTGMIDGWYGYHHQNEQGSGYAADQKSTQNAINGITNKVNSVIEKMNTQFTAVGKEFNNLEKRMENLNKKVDDGFLDIWTYNAELLVLLENERTLDFHDLNVKNLYEKVKSQLKNNAKEIGNGCFEFYHKCDNECMESVRNGTYDYPKYSEESKLNREKIDGVKLESMGVYQILAIYSTVASSLVLLVSLGAISFWMCSNGSLQCRICI',
            'CTV': 'MDDETKKLKNKNKETKEGDDVVAAESSFGSMNLHIDPTLIAMNDVRQLGTQQNAALNRDLFLTLKGKYPNLPDKDKDFHIAMMLYRLAVKSSSLQSDDDTTGITYTREGVEVDLSDKLWTDVVFNSKGIGNRTNALRVWGRTNDALYLAFCRQNRNLSYGGRPLDAGIPAGYHYLCADFLTGAGLTDLECAVYIQAKEQLLKKRGADEVVVTNVRQLGKFNTR',
            'Xanthomonas oryzae': 'MKPAASATPLNRTAPASPAGIHEIEEEASAHASPSHSPAQSEGALTMLSRRPSKRGKETADVTASAAQSASHLQSVELQVSQPAVSPNTGNASLVRLKEQLAADNLRPVEPELAAELINKTRPMKLADATGPQERAATHADLLGRIRETDAMAWYRAQGLSENEVANLRRSALLSGMPNPTGSFLNNAMQYIVSPWINYATHQPWAGAGFGFATAAIAAPMNAAQQSAVVSLCESIREHGGHVIVPDKKQINDKHWLPALAKALESHIAEFSGCCDRFRALKDAADQNPAAQPTADFIAAAHQVLQAETRLHQAQHDFVMTQGAHERQWMGNRWQAVPRILRSPLSGTLGLLSKTGAMRALSPTAQTVGALLMSAVQHVAAGFDEQAKQDYNNKLNLLYADVLTDTGKAKLARGEPVAAEEIDQGKLRKLIQSPTQALVKRITSGLVAMEKELKAQVAAPRSPQATTGDDDLDLEAGHGAGPAKALKLLSQDLKALREGRLDELDPDGVAATLLLGAEKSVVSDQLIGDIIKKYTSREFSAQTAQRIGQMFHLGVLGSAASSVIGKASSAARGGTRNVPIPQALAISALSGGMAAVGALNQHTAITVKNNRREGDTDIGLKQQVLRGVMGGANEALSQRRATKASQAINALVQRSDVEALLSRAKALTQRSGATSSATHASPALTLPEAVEQLRPGVASASQSHEVIVQIGEEDRALPPA'
        }
        self.reference_sequences = reference_sequences or self.default_sequences

    def sequence_to_vector(self, sequence):
        """
        Convert a sequence to a vector representation.
        
        :param sequence: Input DNA/protein sequence
        :return: Numpy vector
        """
        char_counts = Counter(sequence)
        vector = np.array([char_counts.get(chr(i), 0) for i in range(65, 91)])
        return vector.reshape(1, -1)

    def check_sequence_similarity(self, sequence_a, sequence_b):
        """
        Calculate cosine similarity between two sequences.
        
        :param sequence_a: First input sequence
        :param sequence_b: Second input sequence
        :return: Similarity score
        """
        vector_a = self.sequence_to_vector(sequence_a)
        vector_b = self.sequence_to_vector(sequence_b)
        
        similarity = cosine_similarity(vector_a, vector_b)
        return similarity[0][0]

    def analyze_input_sequence(self, input_sequence, threshold=0.85):
        """
        Analyze input sequence against reference sequences.
        
        :param input_sequence: Sequence to compare
        :param threshold: Similarity threshold for model usability
        :return: List of similarity results
        """
        results = []
        for name, seq in tqdm(self.reference_sequences.items(), desc="Analyzing Sequences"):
            similarity_score = self.check_sequence_similarity(input_sequence, seq)
            use_model = "You can use this model" if similarity_score > threshold else "You cannot use this model"
            results.append(f"Similarity score with {name} = {similarity_score:.2f}. {use_model}")
        return results

def read_sequence_from_file(file_path):
    """
    Read a sequence from a file, removing whitespaces and newlines.
    
    :param file_path: Path to the input file
    :return: Cleaned sequence
    """
    try:
        with open(file_path, 'r') as f:
            sequence = f.read().replace('\n', '').replace(' ', '').upper()
        return sequence
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None

def main():
    """
    Main function to run the sequence similarity checker.
    """
    parser = argparse.ArgumentParser(description='Sequence Similarity Checker')
    parser.add_argument('-f', '--file', help='Path to input sequence file')
    args = parser.parse_args()

    # Interactive input if no file is provided
    if not args.file:
        print("No input file provided. Please enter the sequence manually.")
        input_sequence = input("Enter your DNA/Protein sequence: ").upper().replace(' ', '')
    else:
        input_sequence = read_sequence_from_file(args.file)
        if not input_sequence:
            return

    # Validate seq
    if not input_sequence or not all(char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' for char in input_sequence):
        print("Invalid sequence. Please use valid protein/DNA letters.")
        return

    # Initialize and run
    checker = SequenceSimilarityChecker()
    similarity_results = checker.analyze_input_sequence(input_sequence)

    # Print results
    print("\nSimilarity Analysis Results:")
    for result in similarity_results:
        print(result)

if __name__ == "__main__":
    main()
