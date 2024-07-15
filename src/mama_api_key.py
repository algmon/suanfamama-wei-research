import base64
import secrets
import string
from cryptography.fernet import Fernet

def generate_api_key(length=44):
    """Generate a random API key of a specified length.

    Args:
        length (int): The length of the API key. Default is 44 characters.

    Returns:
        str: A randomly generated API key.
    """
    # Exclude ' and " from the character set
    characters = string.ascii_letters + string.digits + ''.join(c for c in string.punctuation if c not in ('"', "'"))
    api_key = ''.join(secrets.choice(characters) for _ in range(length))
    return api_key

# Generate a key for encryption and decryption
# You must store this key securely, as you will need it to decrypt the data
def generate_encryption_key():
    return Fernet.generate_key()

# Function to encrypt the API key
def encrypt_api_key(api_key, encryption_key):
    fernet = Fernet(encryption_key)
    encrypted_api_key = fernet.encrypt(api_key.encode())
    return encrypted_api_key

# Function to decrypt the API key
def decrypt_api_key(encrypted_api_key, encryption_key):
    fernet = Fernet(encryption_key)
    decrypted_api_key = fernet.decrypt(encrypted_api_key).decode()
    return decrypted_api_key

# Function to encode the API key using Base64
def encode_api_key(api_key):
    encoded_api_key = base64.b64encode(api_key.encode()).decode()
    return encoded_api_key

# Function to decode the Base64 encoded API key
def decode_api_key(encoded_api_key):
    decoded_api_key = base64.b64decode(encoded_api_key.encode()).decode()
    return decoded_api_key

def use_mama_api_via_mamam_api_key():
    pass

# Example usage
#api_key = generate_api_key()
#print(f"Generated API key: {api_key}")

# Example usage
#encryption_key = generate_encryption_key()
#print(f"Encryption Key: {encryption_key.decode()}")

#api_key = generate_api_key()
#print(f"Generated API key: {api_key}")

#encrypted_key = encrypt_api_key(api_key, encryption_key)
#print(f"Encrypted API key: {encrypted_key}")

#decrypted_key = decrypt_api_key(encrypted_key, encryption_key)
#print(f"Decrypted API key: {decrypted_key}")

# Example usage
api_key = generate_api_key()
print(f"Generated API key: {api_key}")

encoded_key = encode_api_key(api_key)
print(f"Encoded API key: {encoded_key}")

decoded_key = decode_api_key(encoded_key)
print(f"Decoded API key: {decoded_key}")