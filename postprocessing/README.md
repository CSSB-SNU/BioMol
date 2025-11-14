## Postprocessing Pipeline

**Input:** `CIFMol`

### Steps
0. **Build CIF LMDB**
1. **Extract FASTA**
2. **Generate Sequence Hash**
   - Format: `1 chracter + 7 digits`
   - The first digit represents the molecule type:
     - `P`: Protein (L form)
     - `Q`: Protein (D form)
     - `D`: DNA  
     - `R`: RNA  
     - `N`: Other NA
     - `B`: Branched  
     - `L`: Non-polymer(Ligand)  
     - `X`: Other (e.g., H₂O)
3. **Identify Signal Peptides**
   - Using **SignalP**, build SignalP lmdb (seq_hash -> signal peptide container)
4. **Build A3M LMDB**
5. **Sequence Clustering**
   - **Antibody:** SabDab, CD-HIT, Anarci (H3 if exists, else L3) cluster ID = A1234567 (rep ID, P->A)
   - **Peptides** <10 residues, 100%
   - **Polypeptide(D)** (따로 나눠서)
   - **Other Proteins:** MMseqs2 easy-cluster (seq_id=0.3,cov=0.8,covmode=0,clustmode=1) cluster ID = rep ID
   - **RNA, DNA, Ligands:** 100% sequence identity cluster ID = N1234567 (same as sequence hash)
6. **Graph Clustering**
   - Cluster CIF items using a graph-based algorithm
7. **Build Postprocessed LMDB**
8. **Build DataLoader**

External Source
You have to download it manually.
   - SabDab summary.tsv (https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/summary/all/)
   - Anarci (https://github.com/oxpig/ANARCI.git)
   - CDHit ()
   - MMSeqs2

SabDab parsing
   - SabDab summary.tsv parsing
      You can download it from 
   - seq(.fasta) -> ANARCI (Chotia) -> CDR region
   - CD hit
      # Run CD-HIT on H3 sequences
      /path/to/cd-hit -i /path/to/H3.fasta \
         -o /path/to/cd_hit_H3.dat \
         -c 0.9 -n 2 -M 256000 -d 0 -T 0 -l 1 -s 0.8 -aL 0.8 -aS 0.8 -g 1

      # Run CD-HIT on H3L3 sequences
      /path/to/cd-hit -i /path/to/PDB_2024Mar18/AbAg/H3L3.fasta \
         -o /path/to/cd_hit_H3L3.dat \
         -c 0.9 -n 2 -M 256000 -d 0 -T 0 -l 1 -s 0.8 -aL 0.8 -aS 0.8 -g 1