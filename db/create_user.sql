CREATE USER vector IDENTIFIED BY HelloHello123 DEFAULT TABLESPACE users TEMPORARY TABLESPACE temp QUOTA UNLIMITED ON users;

GRANT CONNECT, RESOURCE TO vector;