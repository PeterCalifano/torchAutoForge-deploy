def Sanitize_matlab_name(name):
    """
    Convert ONNX weight name to MATLAB-compatible field name (max 31 chars)
    """
    # Remove onnx:: prefix
    if name.startswith('onnx::'):
        name = name[6:]

    # Replace problematic characters with underscores
    sanitized = name.replace('.', '_').replace(
        '-', '_').replace('/', '_').replace('\\', '_').replace(':', '_')

    # Ensure it starts with a letter (MATLAB requirement)
    if not sanitized[0].isalpha():
        sanitized = 'w_' + sanitized

    # Truncate to 31 characters (MATLAB limitation)
    if len(sanitized) > 31:
        # Try to keep the end part which is usually more informative (like 'weight', 'bias')
        if '_' in sanitized:
            parts = sanitized.split('_')
            # Keep the last part and truncate the beginning
            last_part = parts[-1]
            remaining_chars = 31 - len(last_part) - 1  # -1 for underscore
            if remaining_chars > 0:
                truncated_start = sanitized[:remaining_chars]
                sanitized = truncated_start + '_' + last_part
            else:
                sanitized = sanitized[:31]
        else:
            sanitized = sanitized[:31]

    return sanitized
