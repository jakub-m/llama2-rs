// info
#[cfg(feature = "log-info")]
#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => {
        {
            eprint!("INFO   ");
            eprintln!($($arg)*);
        }
    };
}

#[cfg(not(feature = "log-info"))]
#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => {{}};
}

// debug
#[cfg(feature = "log-debug")]
#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => {
        {
            eprint!("DEBUG ");
            eprintln!($($arg)*);
        }
    };
}

#[cfg(not(feature = "log-debug"))]
#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => {{}};
}

#[cfg(feature = "log-debug")]
#[macro_export]
macro_rules! debug_print {
    ($($arg:tt)*) => {
        {
            eprint!($($arg)*);
        }
    };
}

#[cfg(not(feature = "log-debug"))]
#[macro_export]
macro_rules! debug_print {
    ($($arg:tt)*) => {{}};
}

// trace
#[cfg(feature = "log-trace")]
#[macro_export]
macro_rules! tracd {
    ($($arg:tt)*) => {
        {
            eprint!("TRACE ");
            eprintln!($($arg)*);
        }
    };
}

#[cfg(not(feature = "log-trace"))]
#[macro_export]
macro_rules! trace {
    ($($arg:tt)*) => {{}};
}
