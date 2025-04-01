import os
import mimetypes
import magic

def identify_file_type(file_path):
    """
    Identify the type of file (image, audio, text, etc.)
    Returns a string representing the file type
    """
    # Initialize mimetypes
    mimetypes.init()
    
    # Try to use python-magic for more accurate detection
    try:
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(file_path)
    except:
        # Fall back to mimetypes if magic fails
        file_type, _ = mimetypes.guess_type(file_path)
    
    if not file_type:
        # Last resort, guess based on extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
            return 'image'
        elif ext in ['.wav', '.mp3', '.flac', '.ogg']:
            return 'audio'
        elif ext in ['.txt', '.md', '.rtf', '.doc', '.docx']:
            return 'text'
        return None
    
    # Extract the main type from the MIME type
    main_type = file_type.split('/')[0]
    
    if main_type == 'image':
        return 'image'
    elif main_type == 'audio':
        return 'audio'
    elif main_type == 'text' or file_type in ['application/pdf', 'application/msword']:
        return 'text'
    
    return None

def read_file_as_bytes(file_path):
    """Read a file and return its contents as bytes"""
    with open(file_path, 'rb') as f:
        return f.read()
