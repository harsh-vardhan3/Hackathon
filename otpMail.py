import random
import yagmail

# Replace with your Zoho Mail credentials
SENDER_EMAIL = "saksham121212@zohomail.in"
SENDER_PASSWORD = "xtcdu8atjh"  # Use an App Password if 2FA is enabled
RECEIVER_EMAIL = "100420vishalsrinivasan.gbkm@gmail.com"

# Temporary storage for OTPs (use Redis/DB in production)
otp_store = {}

def send_email(userEmail, otp):
    try:
        # Update host to `smtp.zoho.in`
        yag = yagmail.SMTP(SENDER_EMAIL, SENDER_PASSWORD, host='smtp.zoho.in', port=465)
        
        subject = "Your OTP for Verification"
        body = f"""
        Hello,

        Your OTP for verification is: {otp}

        Best,
        Your Script
        """
        
        yag.send(userEmail, subject, body)
        print("✅ OTP sent successfully!")
    except Exception as e:
        print("❌ Error sending OTP:", str(e))

def send_otp(userEmail):
    otp = str(random.randint(100000, 999999))  # Generate a 6-digit OTP
    otp_store[userEmail] = otp  # Store OTP temporarily
    
    # Send OTP to the user
    send_email(userEmail, otp)
    return otp

def verify_otp(userEmail, user_otp):
    stored_otp = otp_store.get(userEmail)  # Retrieve stored OTP
    
    if stored_otp and stored_otp == user_otp:
        del otp_store[userEmail]  # Remove OTP after successful verification
        return 1  # OTP verified successfully
    else:
        return 0  # Invalid OTP

# The function will be called when you run the script
userEmail = RECEIVER_EMAIL  # You already have this set to the recipient's email

# Step 1: Send OTP to email
otp_sent = send_otp(userEmail)  # This sends the OTP

# Step 2: Verify OTP
user_otp = input("Enter the OTP received: ")  # Manually input the OTP you received
verification_status = verify_otp(userEmail, user_otp)  # This verifies the OTP

if verification_status == 1:
    print("✅ OTP Verified Successfully!")
else:
    print("❌ Invalid OTP. Please try again.")
