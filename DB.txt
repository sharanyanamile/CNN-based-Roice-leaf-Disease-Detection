create database PlantDiseaseDB;
use PlantDiseaseDB;

create table register(username varchar(30) primary key,
password varchar(30),
contact varchar(12),
email varchar(30),
address varchar(40));

create table locations(username varchar(50),
image_name varchar(50),
predicted_disease varchar(200),
latitude varchar(50),
longitude varchar(50));




