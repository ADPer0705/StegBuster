#!/usr/bin/env python3
"""
Text Steganography Demo Generator for StegBuster
===============================================

This script generates demonstration text files for StegBuster with embedded messages
using different steganography techniques:
1. Whitespace steganography (extra spaces and tabs)
2. Zero-width character steganography
3. Lexical steganography (first letter of each line/word)

The generated files can be used to test the detection capabilities of StegBuster.
"""

import os
import random
import string

# Set random seed for reproducibility
random.seed(42)

# Directory structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_DIR = os.path.join(BASE_DIR, "text")
WHITESPACE_DIR = os.path.join(TEXT_DIR, "whitespace")
ZEROWIDTH_DIR = os.path.join(TEXT_DIR, "zerowidth")
LEXICAL_DIR = os.path.join(TEXT_DIR, "lexical")

# Ensure directories exist
os.makedirs(WHITESPACE_DIR, exist_ok=True)
os.makedirs(ZEROWIDTH_DIR, exist_ok=True)
os.makedirs(LEXICAL_DIR, exist_ok=True)

# Sample secret messages
SHORT_MESSAGE = "StegBuster can find this hidden message! Crypto Finals 2025"
LONG_MESSAGE = """
This is a longer hidden message that demonstrates the capacity of steganography techniques.
StegBuster is designed to detect various forms of steganography across multiple file formats.
This project showcases both traditional statistical methods and machine learning approaches.
Good luck with your Crypto Finals presentation!
"""

# Sample cover text (a small passage about steganography)
COVER_TEXT = """
Steganography is the practice of concealing information within other non-secret data or a physical object to avoid detection. Steganography can be used in combination with encryption as an extra step for hiding or protecting data. The word steganography comes from Greek and literally means "covered writing."

Unlike cryptography, which focuses on making data unreadable without a specific key, steganography is concerned with concealing the fact that secret data exists at all. A message being transmitted may use steganography to disguise its presence in a cover object, such as a digital image file, an audio recording, or a text document.

Digital steganography often involves using sophisticated algorithms to embed data into carrier files. Common techniques include:

1. Least Significant Bit insertion, where data is hidden in the least significant bits of digital files
2. Echo hiding in audio files, where secret messages are incorporated as echo
3. White space manipulation in text documents, where spaces and tabs encode binary data
4. Invisible characters in text, where zero-width characters hide information
5. Metadata manipulation, where file headers and metadata fields store hidden content

Detection of steganography is an active field in computer security known as steganalysis. It involves techniques ranging from statistical analysis to machine learning to identify when a file might contain hidden information.

Modern applications of steganography range from digital watermarking for copyright protection to secure communication between parties who wish to avoid drawing attention to their correspondence. As both steganographic and steganalysis techniques evolve, the field remains an important area of study in information security.
"""

# Additional longer text for lexical steganography
LONGER_TEXT = """
Information hiding through steganography has been practiced for centuries across various cultures and time periods. The earliest recorded uses date back to ancient Greece, where messages were tattooed on messengers' shaved heads, allowing hair to grow back before sending them on their journey. When the messengers arrived, their heads would be shaved again to reveal the hidden message.

Another historical example comes from ancient China, where messages were written on fine silk, which was then compressed into a small ball and covered in wax. The messenger would then swallow the ball to avoid detection during their travels.

During World War II, invisible inks were widely used for covert communication. These inks, often made from organic substances like lemon juice or milk, would remain invisible until exposed to heat, revealing the hidden message to the intended recipient while appearing innocuous to interceptors.

The digital age has transformed steganography into a sophisticated technical discipline. Modern techniques exploit the inherent limitations of human perception, storing information in ways that are difficult for humans to detect without technical assistance. For instance, minor changes to pixel values in an image or slight modifications to audio wave patterns can be imperceptible to human senses but can carry substantial amounts of hidden data.

Various file formats offer different opportunities for concealment. JPEG images use a lossy compression algorithm based on discrete cosine transformations, providing ideal locations to hide data in transformation coefficients. Audio files may hide information in frequency masking areas where human hearing is less sensitive. Text documents can utilize formatting, spacing, or invisible characters for concealment.

The evolution of steganography has been accompanied by the development of steganalysis, which seeks to detect the presence of hidden messages. This cat-and-mouse game continues to drive innovation in both fields, with machine learning and advanced statistical methods now at the forefront of both hiding and finding secret content.

As digital communication becomes increasingly central to modern life, the importance of steganography in both legitimate applications (like digital watermarking for copyright protection) and potential misuse continues to grow. Understanding these techniques is essential for cybersecurity professionals, forensic analysts, and anyone concerned with the full spectrum of information security.
"""

def text_to_binary(text):
    """Convert text to a binary string."""
    binary = ''.join(format(ord(char), '08b') for char in text)
    return binary

def embed_whitespace_steganography(cover_text, secret_message):
    """Embed message using whitespace steganography.
    - One space after period represents 0
    - Two spaces after period represents 1
    
    This version uses a different encoding approach - embedding multiple bits
    at the end of each sentence and paragraph to fit more data in less text.
    """
    binary_message = text_to_binary(secret_message)
    
    # Split the cover text into sentences
    sentences = cover_text.replace('\n\n', '. ').replace('\n', ' ').split('.')
    
    # Remove any empty strings
    sentences = [s for s in sentences if s.strip()]
    
    # Calculate how many bits we can encode
    # We'll use varying numbers of spaces (1-4) to encode 2 bits per sentence
    # 1 space = 00, 2 spaces = 01, 3 spaces = 10, 4 spaces = 11
    max_bits = len(sentences) * 2
    
    if len(binary_message) > max_bits:
        # Truncate the message if it's too long
        print(f"Warning: Message too long, truncating to {max_bits} bits")
        binary_message = binary_message[:max_bits]
    
    # Create the stego text
    stego_text = ""
    for i, sentence in enumerate(sentences):
        if i * 2 < len(binary_message):
            # Get the next two bits (or just one if at the end)
            if i * 2 + 1 < len(binary_message):
                bits = binary_message[i * 2:i * 2 + 2]
            else:
                bits = binary_message[i * 2:i * 2 + 1] + '0'  # Pad with 0
            
            # Add appropriate spacing based on the bits
            if bits == '00':
                stego_text += sentence.strip() + ". "         # 1 space
            elif bits == '01':
                stego_text += sentence.strip() + ".  "        # 2 spaces
            elif bits == '10':
                stego_text += sentence.strip() + ".   "       # 3 spaces
            else:  # '11'
                stego_text += sentence.strip() + ".    "      # 4 spaces
        else:
            # For remaining sentences, just use normal spacing
            stego_text += sentence.strip() + ". "
    
    return stego_text

def embed_zero_width_steganography(cover_text, secret_message):
    """Embed message using zero-width characters.
    - Zero-width space (U+200B) for 0
    - Zero-width non-joiner (U+200C) for 1
    """
    binary_message = text_to_binary(secret_message)
    
    # Zero-width characters
    zero_char = '\u200B'  # Zero-width space
    one_char = '\u200C'   # Zero-width non-joiner
    
    # Convert binary to zero-width string
    zero_width_str = ''
    for bit in binary_message:
        if bit == '0':
            zero_width_str += zero_char
        else:
            zero_width_str += one_char
    
    # Identify a good position to insert the hidden text (after a period)
    parts = cover_text.split('.')
    if len(parts) <= 1:
        # If no periods, insert in the middle
        middle = len(cover_text) // 2
        stego_text = cover_text[:middle] + zero_width_str + cover_text[middle:]
    else:
        # Insert after the first period
        stego_text = parts[0] + '.' + zero_width_str + ''.join('.' + p for p in parts[1:])
    
    return stego_text

def create_lexical_stego_first_letter(secret_message):
    """Create a text where the first letter of each line spells out the secret message."""
    words = []
    
    for char in secret_message.upper():
        if char.isalpha():
            # Find a word starting with this letter
            word_pool = []
            if char.lower() == 'a':
                word_pool = ["Although", "Among", "After", "Additionally", "Across"]
            elif char.lower() == 'b':
                word_pool = ["Because", "Before", "Being", "Between", "Beyond"]
            elif char.lower() == 'c':
                word_pool = ["Consider", "Critically", "Creating", "Consuming", "Computing"]
            elif char.lower() == 'd':
                word_pool = ["During", "Despite", "Demonstrating", "Digital", "Decoding"]
            elif char.lower() == 'e':
                word_pool = ["Every", "Eventually", "Earlier", "Encrypted", "Even"]
            elif char.lower() == 'f':
                word_pool = ["Following", "Furthermore", "Finding", "Finally", "Frequently"]
            elif char.lower() == 'g':
                word_pool = ["Given", "Generally", "Gradually", "Going", "Growing"]
            elif char.lower() == 'h':
                word_pool = ["However", "Hidden", "Having", "Historical", "Helping"]
            elif char.lower() == 'i':
                word_pool = ["Information", "Importantly", "Including", "Investigating", "Invisible"]
            elif char.lower() == 'j':
                word_pool = ["Just", "Judging", "Joining", "Justifying", "Jargon"]
            elif char.lower() == 'k':
                word_pool = ["Knowledge", "Known", "Keeping", "Key", "Kinetic"]
            elif char.lower() == 'l':
                word_pool = ["Looking", "Later", "Learning", "Likewise", "Limiting"]
            elif char.lower() == 'm':
                word_pool = ["Many", "Methods", "Modern", "Machine", "Meanwhile"]
            elif char.lower() == 'n':
                word_pool = ["Notably", "Never", "Numerous", "Nevertheless", "Next"]
            elif char.lower() == 'o':
                word_pool = ["Often", "Only", "Otherwise", "Observing", "Obtaining"]
            elif char.lower() == 'p':
                word_pool = ["Perhaps", "Previously", "Processing", "Primarily", "Preparing"]
            elif char.lower() == 'q':
                word_pool = ["Quite", "Quickly", "Questioning", "Quality", "Quantifying"]
            elif char.lower() == 'r':
                word_pool = ["Rather", "Recently", "Regarding", "Revealing", "Researchers"]
            elif char.lower() == 's':
                word_pool = ["Security", "Steganography", "Several", "Specifically", "Subsequently"]
            elif char.lower() == 't':
                word_pool = ["Therefore", "Typically", "Throughout", "Techniques", "Today"]
            elif char.lower() == 'u':
                word_pool = ["Understanding", "Usually", "Under", "Using", "Ultimately"]
            elif char.lower() == 'v':
                word_pool = ["Various", "Virtually", "Very", "Visualizing", "Verifying"]
            elif char.lower() == 'w':
                word_pool = ["While", "When", "With", "Without", "Whether"]
            elif char.lower() == 'x':
                word_pool = ["Xerox", "X-rays", "XML", "Xenon", "Xylograph"]  # X is hard!
            elif char.lower() == 'y':
                word_pool = ["Yet", "Years", "Yielding", "Yesterday", "You"]
            elif char.lower() == 'z':
                word_pool = ["Zero", "Zealous", "Zones", "Zenith", "Zooming"]
            
            # Choose a random word from the pool
            if word_pool:
                selected_word = random.choice(word_pool)
                # Generate a random sentence starting with this word
                sentence = selected_word + " " + ' '.join(random.sample([
                    "the", "this", "these", "those", "some", "many", "information", "data", 
                    "files", "techniques", "methods", "systems", "security", "encryption", 
                    "detection", "algorithms", "research", "analysis", "development", 
                    "implementation", "process", "strategy", "approach"
                ], 3 + random.randint(0, 7)))
                
                # Add some variety to sentence endings
                endings = [
                    "is an important consideration.", 
                    "remains a key focus.", 
                    "provides valuable insights.", 
                    "deserves further study.",
                    "shows promising results.", 
                    "continues to evolve.", 
                    "presents unique challenges.",
                    "offers new opportunities.", 
                    "requires careful attention."
                ]
                
                sentence += " " + random.choice(endings)
                words.append(sentence)
        elif char == ' ':
            # For spaces in the message, add a short paragraph break
            words.append("")
    
    return '\n'.join(words)

def main():
    print("Generating steganography demo text files...")
    
    # === Whitespace Steganography ===
    print("Creating whitespace steganography examples...")
    
    # Example with short message
    stego_whitespace_short = embed_whitespace_steganography(COVER_TEXT, SHORT_MESSAGE)
    with open(os.path.join(WHITESPACE_DIR, "whitespace_short.txt"), 'w', encoding='utf-8') as f:
        f.write(stego_whitespace_short)
    
    # Example with clean text for comparison
    with open(os.path.join(WHITESPACE_DIR, "original.txt"), 'w', encoding='utf-8') as f:
        f.write(COVER_TEXT.replace('\n', ' ').replace('.  ', '. ').replace('.   ', '. '))
    
    # === Zero-width Character Steganography ===
    print("Creating zero-width steganography examples...")
    
    # Example with short message
    stego_zw_short = embed_zero_width_steganography(COVER_TEXT, SHORT_MESSAGE)
    with open(os.path.join(ZEROWIDTH_DIR, "zerowidth_short.txt"), 'w', encoding='utf-8') as f:
        f.write(stego_zw_short)
    
    # Example with longer message
    stego_zw_long = embed_zero_width_steganography(LONGER_TEXT, LONG_MESSAGE[:100])
    with open(os.path.join(ZEROWIDTH_DIR, "zerowidth_long.txt"), 'w', encoding='utf-8') as f:
        f.write(stego_zw_long)
    
    # Example with clean text for comparison
    with open(os.path.join(ZEROWIDTH_DIR, "original.txt"), 'w', encoding='utf-8') as f:
        f.write(LONGER_TEXT)
    
    # === Lexical Steganography ===
    print("Creating lexical steganography examples...")
    
    # Example with first letter of each line spelling the message
    lexical_stego = create_lexical_stego_first_letter(SHORT_MESSAGE[:20])  # Use first 20 chars
    with open(os.path.join(LEXICAL_DIR, "lexical_first_letter.txt"), 'w', encoding='utf-8') as f:
        f.write(lexical_stego)
    
    # Create a README explaining the steganography
    with open(os.path.join(LEXICAL_DIR, "README.txt"), 'w', encoding='utf-8') as f:
        f.write("Lexical Steganography Examples\n")
        f.write("============================\n\n")
        f.write("In lexical_first_letter.txt, the first letter of each line spells out a secret message.\n")
        f.write("This is a form of acrostic steganography.\n\n")
        f.write("The hidden message is: " + SHORT_MESSAGE[:20] + "\n")
    
    print("Text steganography demo files generated successfully!")
    print(f"Whitespace examples saved to: {WHITESPACE_DIR}")
    print(f"Zero-width examples saved to: {ZEROWIDTH_DIR}")
    print(f"Lexical examples saved to: {LEXICAL_DIR}")

if __name__ == "__main__":
    main()