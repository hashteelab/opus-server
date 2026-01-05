import logging
import sys


class CleanFormatter(logging.Formatter):
    """Custom formatter that removes the INFO:op_schemes: prefix for cleaner output"""
    
    def format(self, record):
        # Only include the message, not the logger name or level
        return record.getMessage()


# Configure logging with custom formatter
def setup_logger():
    """Setup logger with clean formatting"""
    logger = logging.getLogger("op_schemes")
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Create console handler with custom formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(CleanFormatter())
    
    logger.addHandler(handler)
    logger.propagate = False
    
    return logger


# Initialize logger
logger = setup_logger()


# ============================================================================
# HELPER FUNCTIONS FOR STRUCTURED LOGGING
# ============================================================================

def log_header(title: str, char: str = "="):
    """
    Log a major section header with heavy border
    
    Args:
        title: Section title
        char: Border character (default: =)
    """
    border = char * 80
    logger.info("")
    logger.info(border)
    logger.info(f"{title.center(80)}")
    logger.info(border)
    logger.info("")


def log_section(title: str, char: str = "-"):
    """
    Log a subsection header with lighter border
    
    Args:
        title: Section title
        char: Border character (default: -)
    """
    border = char * 80
    logger.info("")
    logger.info(border)
    logger.info(title)
    logger.info(border)


def log_step(step_number: int, description: str):
    """
    Log a step in the process with clear formatting
    
    Args:
        step_number: Step number
        description: Step description
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"STEP {step_number}: {description}".center(80))
    logger.info("=" * 80)
    logger.info("")


def log_subsection(title: str):
    """
    Log a subsection with simple separator
    
    Args:
        title: Subsection title
    """
    logger.info("")
    logger.info(f"{'─' * 80}")
    logger.info(f"  {title}")
    logger.info(f"{'─' * 80}")


def log_item(label: str, value: str = "", indent: int = 2):
    """
    Log an item with label and value
    
    Args:
        label: Item label
        value: Item value (optional)
        indent: Number of spaces to indent (default: 2)
    """
    indent_str = " " * indent
    if value:
        logger.info(f"{indent_str}• {label}: {value}")
    else:
        logger.info(f"{indent_str}• {label}")


def log_success(message: str, indent: int = 2):
    """
    Log a success message with checkmark
    
    Args:
        message: Success message
        indent: Number of spaces to indent (default: 2)
    """
    indent_str = " " * indent
    logger.info(f"{indent_str}✓ {message}")


def log_error(message: str, indent: int = 2):
    """
    Log an error message with X mark
    
    Args:
        message: Error message
        indent: Number of spaces to indent (default: 2)
    """
    indent_str = " " * indent
    logger.error(f"{indent_str}✗ {message}")


def log_warning(message: str, indent: int = 2):
    """
    Log a warning message with warning symbol
    
    Args:
        message: Warning message
        indent: Number of spaces to indent (default: 2)
    """
    indent_str = " " * indent
    logger.warning(f"{indent_str}⚠ {message}")


def log_info(message: str, indent: int = 2):
    """
    Log an info message with bullet point
    
    Args:
        message: Info message
        indent: Number of spaces to indent (default: 2)
    """
    indent_str = " " * indent
    logger.info(f"{indent_str}{message}")


def log_processing(message: str, indent: int = 2):
    """
    Log a processing message with spinner/arrow
    
    Args:
        message: Processing message
        indent: Number of spaces to indent (default: 2)
    """
    indent_str = " " * indent
    logger.info(f"{indent_str}→ {message}")


def log_data_point(label: str, value, indent: int = 4):
    """
    Log a data point (key-value pair)
    
    Args:
        label: Data label
        value: Data value
        indent: Number of spaces to indent (default: 4)
    """
    indent_str = " " * indent
    logger.info(f"{indent_str}{label}: {value}")


def log_list_item(item: str, indent: int = 4):
    """
    Log a list item with dash
    
    Args:
        item: List item text
        indent: Number of spaces to indent (default: 4)
    """
    indent_str = " " * indent
    logger.info(f"{indent_str}- {item}")


def log_separator(char: str = "-", width: int = 80):
    """
    Log a simple separator line
    
    Args:
        char: Character to use for separator
        width: Width of separator line
    """
    logger.info(char * width)


def log_empty_line():
    """Log an empty line for spacing"""
    logger.info("")
